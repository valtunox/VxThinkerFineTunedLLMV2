"""
VaLLM Specialist Model - Training Script.

Author: Joel Otepa Wembo
https://joelwembo.com

Fine-tunes a causal LLM on IT-specialist CSV data and documents from curated
folders under app/data/datasets/ and exports the trained model to
app/data/models/model/. Location: app/services/ai/ml/train.py.

By default the script loads only IT-focused data from:
  - app/data/datasets/cloud/
  - app/data/datasets/automation/
  - app/data/datasets/customer_service/
  - app/data/datasets/skills/
  - app/data/datasets/uploaded/

QUICK START (run from project root directory):
================================================

    # 1. Precompute embeddings (includes documents)
    python -m app.services.ai.ml.precompute

    # 2. Train with the default model (includes documents)
    python -m app.services.ai.ml.train --num-train-epochs 1

    # 3. Start the FastAPI server
    python -m app.app

ADVANCED OPTIONS:
=================

    # Train with a specific model
    python -m app.services.ai.ml.train --model-name-or-path microsoft/phi-2

    # Train for more epochs
    python -m app.services.ai.ml.train --num-train-epochs 3

    # Train on CSVs only (skip documents)
    python -m app.services.ai.ml.train --no-documents

    # Train on a specific CSV only
    python -m app.services.ai.ml.train --dataset app/data/datasets/data.csv --dataset-dir "" --no-documents

OUTPUT:
=======
    app/data/models/model/
        config.json
        tokenizer.json
        pytorch_model.bin
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys
import io

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch

try:
    from .specialist_profile import (
        SPECIALIST_PRIMARY_DATASET,
        SPECIALIST_SYSTEM_PROMPT,
        allowed_dataset_group_names,
        dataset_group_for_path,
        filter_specialist_csv_paths,
        filter_specialist_document_paths,
        specialist_source_label,
    )
except ImportError:
    from app.services.ai.ml.specialist_profile import (
        SPECIALIST_PRIMARY_DATASET,
        SPECIALIST_SYSTEM_PROMPT,
        allowed_dataset_group_names,
        dataset_group_for_path,
        filter_specialist_csv_paths,
        filter_specialist_document_paths,
        specialist_source_label,
    )

# ============================================================================
# HARDCODED MODEL CONFIGURATION (causal LM for generation / analysis only)
# ============================================================================
# This script trains a CAUSAL LM (GPT-style). It is NOT used by AI Matching.
# Matching uses: EmbeddingService (BGE) + precompute FAISS + cross-encoder (ms-marco).
#
# Choose ONE causal LM based on your hardware:
#
# FOR CPU / LOW VRAM (< 8GB) - Fast local testing:
LLM_MODEL_NAME = "distilgpt2"   # ~80M params, runs on CPU
# FOR MEDIUM GPU (8-12GB VRAM):
# LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # 1.1B params, ~3GB VRAM
# LLM_MODEL_NAME = "microsoft/phi-2"                    # 2.7B params, ~6GB VRAM
#
# FOR HIGH-END GPU (16GB+ VRAM):
# LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # 7B params, ~14GB VRAM
# ============================================================================
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


@dataclass
class TrainConfig:
    dataset_path: Path
    dataset_dir: Optional[Path]
    primary_csv: str
    model_name_or_path: str
    output_dir: Path
    text_max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    seed: int
    limit_rows: int
    no_documents: bool = False


def row_to_text(row: Dict) -> str:
    """Convert one CSV row into a single training text.

    For the IT specialist model, a compact key/value representation works well.
    """
    parts: List[str] = []
    for k, v in row.items():
        if v is None:
            continue
        sv = str(v)
        if sv.strip() == "" or sv.strip().lower() == "nan":
            continue
        parts.append(f"{k}: {sv}")

    # A simple instruction prefix encourages instruction-following behavior.
    return (
        SPECIALIST_SYSTEM_PROMPT
        + "\n\nAnalyze the following IT or support record and provide a practical answer.\n\n"
        + " | ".join(parts)
        + "\n\nAnswer:"
    )


def _load_document_text(file_path: Path) -> Optional[str]:
    """Extract plain text from a supported document file."""
    suffix = file_path.suffix.lower()

    try:
        if suffix == '.txt' or suffix == '.md':
            return file_path.read_text(encoding='utf-8', errors='ignore').strip()

        if suffix == '.pdf':
            try:
                import pdfplumber
                with pdfplumber.open(str(file_path)) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                return text.strip()
            except Exception:
                import pypdf
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    return "\n".join(p.extract_text() or "" for p in reader.pages).strip()

        if suffix in ('.docx', '.doc'):
            from docx import Document as DocxDocument
            doc = DocxDocument(str(file_path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()

    except Exception as e:
        print(f"    ⚠️  Could not read {file_path.name}: {e}")

    return None


def doc_to_training_text(text: str, source_type: str, file_name: str) -> str:
    """Wrap raw document text in an instruction-style training prompt."""
    source_label = specialist_source_label(source_type)
    return (
        SPECIALIST_SYSTEM_PROMPT
        + f"\n\nAnalyze the following {source_label} content and provide IT guidance.\n\n"
        f"Source: {source_label} - {file_name}\n\n"
        f"{text}\n\nAnswer:"
    )


def _legacy_load_documents_as_dataframe(data_dir: Path) -> pd.DataFrame:
    """Load IT-specialist documents into a DataFrame with a 'training_text' column.

    The resulting DataFrame has a single column so it can be concatenated with
    CSV frames and processed through the same training pipeline.
    """
    doc_dirs = {
        group_name: data_dir / group_name
        for group_name in allowed_dataset_group_names()
    }

    rows: List[Dict] = []

    for dir_name, dir_path in doc_dirs.items():
        if not dir_path.exists():
            print(f"  ⚠️  Document directory not found: {dir_path}")
            continue

        doc_files = []
        for ext in ['.pdf', '.docx', '.doc', '.txt', '.md']:
            doc_files.extend(dir_path.rglob(f'*{ext}'))

        if not doc_files:
            print(f"  ⚠️  No documents in {dir_name}/")
            continue

        print(f"\n  📂 Loading {len(doc_files)} file(s) from {dir_name}/")
        for doc_file in sorted(doc_files):
            text = _load_document_text(doc_file)
            if not text:
                continue
            rows.append({
                'training_text': doc_to_training_text(text, dir_name, doc_file.stem),
                'source': dir_name,
                'file_name': doc_file.name,
            })
            print(f"    ✓ {doc_file.name} ({len(text)} chars)")

    if rows:
        print(f"\n  ✅ Loaded {len(rows)} documents for training")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_documents_as_dataframe(data_dir: Path) -> pd.DataFrame:
    """Load only IT-specialist documents into a DataFrame."""
    doc_dirs = {
        group_name: data_dir / group_name
        for group_name in allowed_dataset_group_names()
    }

    rows: List[Dict] = []

    for dir_name, dir_path in doc_dirs.items():
        if not dir_path.exists():
            continue

        doc_files: List[Path] = []
        for ext in [".pdf", ".docx", ".doc", ".txt", ".md", ".html"]:
            doc_files.extend(dir_path.rglob(f"*{ext}"))
        doc_files = filter_specialist_document_paths(doc_files, data_dir)

        if not doc_files:
            continue

        label = specialist_source_label(dir_name)
        print(f"\n  Loading {len(doc_files)} file(s) from {dir_name}/ ({label})")
        for doc_file in sorted(doc_files):
            text = _load_document_text(doc_file)
            if not text:
                continue
            rows.append(
                {
                    "training_text": doc_to_training_text(text, dir_name, doc_file.stem),
                    "source": dir_name,
                    "file_name": doc_file.name,
                }
            )
            print(f"    Loaded {doc_file.name} ({len(text)} chars)")

    if rows:
        print(f"\n  Loaded {len(rows)} IT documents for training")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


class CSVCausalLMDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx].to_dict()

        # Rows from documents already carry a ready-made prompt in 'training_text'
        if 'training_text' in row and row.get('training_text'):
            text = str(row['training_text'])
        else:
            text = row_to_text(row)

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item


def keep_only_required_model_files(model_dir: Path) -> None:
    """Ensure the output folder contains exactly:

    /model
      config.json
      tokenizer.json
      pytorch_model.bin

    HuggingFace sometimes writes extra files (generation_config.json, tokenizer_config.json, etc.).
    This function removes extras after saving.

    NOTE: Using a fast tokenizer, `tokenizer.json` is self-contained.
    """

    required = {"config.json", "tokenizer.json", "pytorch_model.bin", "tokenizer_config.json", "special_tokens_map.json"}

    for p in model_dir.iterdir():
        if p.is_file() and p.name not in required:
            p.unlink()


def ensure_pytorch_bin_weights(model_dir: Path) -> None:
    """Ensure weights are stored in exactly `pytorch_model.bin`.

    Depending on environment, Transformers may write `model.safetensors` instead.
    This converts it to `pytorch_model.bin` when possible.
    """

    pytorch_path = model_dir / "pytorch_model.bin"
    safetensors_path = model_dir / "model.safetensors"

    if pytorch_path.exists():
        return

    if not safetensors_path.exists():
        return

    # Convert safetensors -> pytorch_model.bin
    try:
        from safetensors.torch import load_file
    except Exception as e:
        raise RuntimeError(
            "Found model.safetensors but could not import safetensors. "
            "Install `safetensors` or ensure Transformers saves .bin weights."
        ) from e

    state_dict = load_file(str(safetensors_path))
    torch.save(state_dict, str(pytorch_path))
    safetensors_path.unlink()


def _fmt_duration(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def _model_param_count(model) -> str:
    """Return a human-readable parameter count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total >= 1e9:
        return f"{total/1e9:.1f}B total, {trainable/1e9:.1f}B trainable"
    if total >= 1e6:
        return f"{total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable"
    return f"{total:,} total, {trainable:,} trainable"


class ProgressPrinter(TrainerCallback):
    """Prints live training metrics on every log step."""

    def __init__(self, total_steps: int, train_start: float):
        self.total_steps = total_steps
        self.train_start = train_start
        self.best_loss = float("inf")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch")
        step = state.global_step
        elapsed = time.time() - self.train_start

        pct = (step / self.total_steps * 100) if self.total_steps > 0 else 0
        parts = [f"step {step}/{self.total_steps} ({pct:.0f}%)"]
        if epoch is not None:
            parts.append(f"epoch={epoch:.2f}")
        if loss is not None:
            marker = ""
            if loss < self.best_loss:
                self.best_loss = loss
                marker = " *best*"
            parts.append(f"loss={loss:.4f}{marker}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        parts.append(f"elapsed={_fmt_duration(elapsed)}")

        if step > 0 and self.total_steps > step:
            eta = elapsed / step * (self.total_steps - step)
            parts.append(f"eta={_fmt_duration(eta)}")

        print("  🟢 " + " | ".join(parts), flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        print("\n  ⏳ Training loop started...", flush=True)

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = state.epoch or 0
        print(f"\n  📗 Epoch {int(epoch) + 1}/{int(args.num_train_epochs)} starting...", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch or 0
        elapsed = time.time() - self.train_start
        print(
            f"  📘 Epoch {int(epoch)}/{int(args.num_train_epochs)} complete "
            f"| elapsed={_fmt_duration(elapsed)}",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.train_start
        print(f"\n  ✅ Training loop finished in {_fmt_duration(elapsed)}", flush=True)


def train(cfg: TrainConfig) -> None:
    global_start = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 80)
    print("🚀 VaLLM - LLM TRAINING")
    print("=" * 80)
    print(f"  📅 Started       : {now}")
    print(f"  🧠 Model         : {cfg.model_name_or_path}")
    print(f"  📂 Output        : {cfg.output_dir}")
    print(f"  🎲 Seed          : {cfg.seed}")
    print("=" * 80)

    set_seed(cfg.seed)

    # ── STEP 1: Load dataset ────────────────────────────────────────────────
    step_start = time.time()
    print("\n" + "-" * 70)
    print("📁 STEP 1/5: Loading datasets")
    print("-" * 70)

    if cfg.dataset_dir is not None:
        if not cfg.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {cfg.dataset_dir}")

        csv_paths = filter_specialist_csv_paths(list(cfg.dataset_dir.rglob("*.csv")), cfg.dataset_dir)
        if not csv_paths:
            raise FileNotFoundError(
                f"No IT-specialist CSV files found in: {cfg.dataset_dir}. "
                f"Expected folders: {', '.join(allowed_dataset_group_names())}"
            )

        primary = cfg.dataset_dir / cfg.primary_csv
        if primary.exists() and dataset_group_for_path(primary, cfg.dataset_dir):
            csv_paths = [primary] + [p for p in csv_paths if p.resolve() != primary.resolve()]

        print(f"  📂 Directory : {cfg.dataset_dir}")
        print(f"  📊 CSV files : {len(csv_paths)}")
        print(f"  Scope         : {', '.join(allowed_dataset_group_names())}")
        print()

        frames = []
        total_csv_rows = 0
        for idx, p in enumerate(csv_paths, 1):
            try:
                frame = pd.read_csv(p, on_bad_lines='warn')
                try:
                    relative_path = str(p.resolve().relative_to(cfg.dataset_dir.resolve())).replace("\\", "/")
                except Exception:
                    relative_path = p.name
                frame["_dataset_file"] = relative_path
                frame["_dataset_group"] = dataset_group_for_path(p, cfg.dataset_dir) or "unknown"
                row_count = len(frame)
                col_count = len(frame.columns)
                total_csv_rows += row_count
                frames.append(frame)
                is_primary = ""
                try:
                    is_primary = " (primary)" if p.resolve() == primary.resolve() else ""
                except Exception:
                    is_primary = ""
                print(f"  [{idx}/{len(csv_paths)}] ✓ {p.name}{is_primary}")
                print(f"           {row_count:,} rows x {col_count} columns")
            except Exception as e:
                print(f"  [{idx}/{len(csv_paths)}] ⚠️  {p.name}: {e}")
                continue

        if not frames:
            raise FileNotFoundError(f"No valid CSV files could be parsed in: {cfg.dataset_dir}")

        df = pd.concat(frames, ignore_index=True)
        dataset_label = str(cfg.dataset_dir)
        print(f"\n  📊 CSV subtotal: {total_csv_rows:,} rows from {len(frames)} file(s)")
    else:
        if not cfg.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {cfg.dataset_path}")
        print(f"  📄 Single file: {cfg.dataset_path}")
        df = pd.read_csv(cfg.dataset_path, on_bad_lines='warn')
        print(f"  ✓ Loaded {len(df):,} rows x {len(df.columns)} columns")
        dataset_label = str(cfg.dataset_path)

    if len(df) == 0:
        raise ValueError(f"Dataset is empty: {dataset_label}")

    step_elapsed = time.time() - step_start
    print(f"  ⏱️  CSV loading completed in {_fmt_duration(step_elapsed)}")

    # ── STEP 1b: Load documents (financial, business) ──────────────────────
    if not cfg.no_documents:
        doc_start = time.time()
        doc_data_dir = cfg.dataset_dir if cfg.dataset_dir else cfg.dataset_path.parent
        print(f"\n  📁 Loading documents from {doc_data_dir}...")
        doc_df = load_documents_as_dataframe(doc_data_dir)
        if len(doc_df) > 0:
            df = pd.concat([df, doc_df], ignore_index=True)
            print(f"\n  📊 Combined dataset: {len(df):,} rows (CSVs + {len(doc_df)} documents)")
        else:
            print("  ⚠️  No documents loaded; training on CSVs only")
        doc_elapsed = time.time() - doc_start
        print(f"  ⏱️  Document loading completed in {_fmt_duration(doc_elapsed)}")
    else:
        print("\n  ⏭️  Skipping document loading (--no-documents)")

    if cfg.limit_rows and cfg.limit_rows > 0:
        df = df.head(cfg.limit_rows).reset_index(drop=True)
        print(f"  🔒 Limited to first {cfg.limit_rows} rows (--limit-rows)")

    # ── STEP 2: Load model & tokenizer ──────────────────────────────────────
    step_start = time.time()
    print("\n" + "-" * 70)
    print("🧠 STEP 2/5: Loading model & tokenizer")
    print("-" * 70)
    print(f"  📦 Model     : {cfg.model_name_or_path}")
    print(f"  ⏳ Downloading / loading from cache...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    step_elapsed = time.time() - step_start
    print(f"  ✅ Model loaded in {_fmt_duration(step_elapsed)}")
    print(f"  🔢 Parameters: {_model_param_count(model)}")
    print(f"  📝 Vocab size: {tokenizer.vocab_size:,}")
    print(f"  📏 Max length: {cfg.text_max_length}")

    # ── STEP 3: Prepare dataset ─────────────────────────────────────────────
    step_start = time.time()
    print("\n" + "-" * 70)
    print("📝 STEP 3/5: Preparing tokenized dataset")
    print("-" * 70)

    dataset = CSVCausalLMDataset(df=df, tokenizer=tokenizer, max_length=cfg.text_max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    step_elapsed = time.time() - step_start
    print(f"  ✅ Dataset ready: {len(dataset):,} samples")
    print(f"  ⏱️  Prepared in {_fmt_duration(step_elapsed)}")

    # ── STEP 4: Configure & run training ────────────────────────────────────
    print("\n" + "-" * 70)
    print("⚙️  STEP 4/5: Training")
    print("-" * 70)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    fp16 = bool(use_cuda and not bf16)

    device_label = "CPU"
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        device_label = f"CUDA - {gpu_name} ({gpu_mem:.1f} GB)"

    device_count = torch.cuda.device_count() if use_cuda else 1
    effective_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * device_count
    steps_per_epoch = math.ceil(len(dataset) / (cfg.per_device_train_batch_size * device_count))
    total_opt_steps = max(1, math.ceil(steps_per_epoch / cfg.gradient_accumulation_steps))
    total_steps = int(total_opt_steps * cfg.num_train_epochs)
    log_every = max(1, total_opt_steps // 10)

    print(f"  💻 Device              : {device_label}")
    print(f"     Precision           : {'bf16' if bf16 else 'fp16' if fp16 else 'fp32'}")
    print(f"     Device count        : {device_count}")
    print(f"  📊 Training samples    : {len(dataset):,}")
    print(f"     Batch size/device   : {cfg.per_device_train_batch_size}")
    print(f"     Gradient accum      : {cfg.gradient_accumulation_steps}")
    print(f"     Effective batch     : {effective_batch}")
    print(f"  🔄 Epochs              : {cfg.num_train_epochs}")
    print(f"     Steps/epoch         : ~{steps_per_epoch:,}")
    print(f"     Optimizer steps     : ~{total_steps:,}")
    print(f"     Log every           : {log_every} steps")
    print(f"  📈 Learning rate       : {cfg.learning_rate:.1e}")
    print(f"     Weight decay        : {cfg.weight_decay}")
    print(f"     Warmup ratio        : {cfg.warmup_ratio}")

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir / "_checkpoints"),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=max(1, int(total_steps * cfg.warmup_ratio)),
        logging_strategy="steps",
        logging_steps=log_every,
        logging_first_step=True,
        save_steps=500,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=0,
        disable_tqdm=False,
        report_to=[],
    )

    train_start = time.time()
    progress_cb = ProgressPrinter(total_steps=total_steps, train_start=train_start)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[progress_cb],
    )

    trainer.train()
    train_elapsed = time.time() - train_start

    # ── STEP 5: Save model ──────────────────────────────────────────────────
    step_start = time.time()
    print("\n" + "-" * 70)
    print("💾 STEP 5/5: Saving model")
    print("-" * 70)
    print(f"  📂 Output directory: {cfg.output_dir}")

    trainer.model.save_pretrained(cfg.output_dir)
    print("  ✓ Model weights saved")

    tokenizer.save_pretrained(cfg.output_dir)
    print("  ✓ Tokenizer saved")

    ensure_pytorch_bin_weights(cfg.output_dir)
    print("  ✓ Weights converted to pytorch_model.bin")

    keep_only_required_model_files(cfg.output_dir)
    print("  ✓ Cleaned up extra files")

    required_paths = [
        cfg.output_dir / "config.json",
        cfg.output_dir / "tokenizer.json",
        cfg.output_dir / "pytorch_model.bin",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"Model export incomplete; missing files: {missing}")

    save_elapsed = time.time() - step_start
    print(f"  ⏱️  Model saved in {_fmt_duration(save_elapsed)}")

    # ── Final files listing ─────────────────────────────────────────────────
    print("\n  📂 Output files:")
    for p in sorted(cfg.output_dir.iterdir()):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"     {p.name:30s} {size_mb:>8.2f} MB")

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - global_start
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"""
    📊 Summary
    {'─' * 45}
    🧠 Model            : {cfg.model_name_or_path}
    🔢 Parameters       : {_model_param_count(model)}
    📝 Training samples : {len(dataset):,}
    🔄 Epochs           : {cfg.num_train_epochs}
    📉 Final best loss  : {progress_cb.best_loss:.4f}
    💻 Device           : {device_label}
    ⏱️  Training time   : {_fmt_duration(train_elapsed)}
    ⏱️  Total time      : {_fmt_duration(total_elapsed)}

    📂 Output
    {'─' * 45}
    📁 Directory        : {cfg.output_dir}
    📄 config.json      : ✅
    📄 tokenizer.json   : ✅
    📄 pytorch_model.bin : ✅

    ✅ Ready! Run 'python -m app.app' to start the server.
    """)


def _app_dir() -> Path:
    """App directory (app/data/...). train.py is in app/services/ai/ml/."""
    return Path(__file__).resolve().parents[3]


def parse_args() -> TrainConfig:
    _app = _app_dir()
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM on IT-specialist data from app/data/datasets and output to app/data/models/model"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(_app / "data" / "datasets" / Path(SPECIALIST_PRIMARY_DATASET)),
        help="Path to training CSV",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        nargs="?",
        const="",
        default=str(_app / "data" / "datasets"),
        help="If set, train on IT-specialist CSVs in this folder (recommended)",
    )
    parser.add_argument(
        "--primary-csv",
        type=str,
        default=SPECIALIST_PRIMARY_DATASET,
        help="CSV to prioritize first when training on --dataset-dir",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=LLM_MODEL_NAME,
        help=f"HuggingFace model name/path (causal LM). Default: {LLM_MODEL_NAME}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_app / "data" / "models" / "model"),
        help="Output folder (app/data/models/model)",
    )
    parser.add_argument("--text-max-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="If > 0, only use the first N rows from the loaded dataset (useful for quick tests)",
    )
    parser.add_argument(
        "--no-documents",
        action="store_true",
        default=False,
        help="Skip loading documents (train on CSVs only)",
    )

    args = parser.parse_args()

    return TrainConfig(
        dataset_path=Path(args.dataset),
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        primary_csv=args.primary_csv,
        model_name_or_path=args.model_name_or_path,
        output_dir=Path(args.output_dir),
        text_max_length=args.text_max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        limit_rows=args.limit_rows,
        no_documents=args.no_documents,
    )


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()

