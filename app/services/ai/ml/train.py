"""
VaLLM Training Script - Fine-tune a Causal LLM on Cloud Operations Data

This script trains a HuggingFace causal language model on multi-format data from app/data/datasets/
Supported formats: CSV, JSON, TXT, PDF

QUICK START (run from va_llm_v1 root directory):
================================================

    # 1. Precompute embeddings (uses all-MiniLM-L6-v2, small and works on CPU)
    python ./app/precompute.py

    # 2. Train with the default model (distilgpt2 - works on CPU!)
    python ./app/train.py --num-train-epochs 1

    # 3. Start the FastAPI server
    python -m app.app

ADVANCED OPTIONS:
=================

    # Train with a specific model
    python ./app/train.py --model-name-or-path microsoft/phi-2

    # Train for more epochs
    python ./app/train.py --num-train-epochs 3

    # Train on specific file types
    python ./app/train.py --file-types csv,json,txt,pdf

    # Train on a specific file only
    python ./app/train.py --dataset ./app/data/datasets/cloud_deployments.csv --dataset-dir
    python ./app/train.py --dataset ./app/data/datasets/deployments.json --dataset-dir
    python ./app/train.py --dataset ./app/data/datasets/cloud_operations_provisionning_knowledge1.txt --dataset-dir

SUPPORTED FILE FORMATS:
=======================
    - CSV:  Tabular data with headers (converted row-by-row)
    - JSON: Arrays of objects, nested structures, or OpenTelemetry-style data
    - TXT:  Plain text files (chunked into paragraphs/sections)
    - PDF:  PDF documents (page-by-page extraction)

OUTPUT:
=======
    app/data/models/
        config.json
        tokenizer.json
        pytorch_model.bin
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

try:
    import PyPDF2
    HAVE_PDF = True
except ImportError:
    HAVE_PDF = False

# ============================================================================
# HARDCODED MODEL CONFIGURATION
# ============================================================================
# Choose ONE model based on your hardware:
#
# ULTRA TINY (< 50MB) - Fastest, great for testing:
# LLM_MODEL_NAME = "sshleifer/tiny-gpt2"                  # ~2MB, instant on CPU
# LLM_MODEL_NAME = "roneneldan/TinyStories-1M"          # ~4MB, 1M params
LLM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"          # ~32MB, 8M params
# LLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"       # ~15GB VRAM
# LLM_MODEL_NAME = "roneneldan/TinyStories-33M"         # ~130MB, 33M params
#
# FOR CPU / LOW VRAM (< 8GB) - Good quality:
# LLM_MODEL_NAME = "distilgpt2"                         # ~350MB, runs on CPU
# FOR MEDIUM GPU (8-12GB VRAM):
# LLM_MODEL_NAME = "microsoft/phi-2"                    # 2.7B params, ~6GB VRAM
# LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # 1.1B params, ~3GB VRAM
## FOR RESUMES & REASONING:
# LLM_MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"  # ~16GB VRAM
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
    file_types: List[str]  # Supported: csv, json, txt, pdf


def row_to_text(row: Dict) -> str:
    """Convert one CSV row into a single training text.

    For deployment/provisioning JSON, a compact key/value representation works well.
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
        "You are an AI assistant for cloud provisioning and deployment. "
        "Analyze the following telemetry record and provide insights.\n\n"
        + " | ".join(parts)
        + "\n\nAnswer:"
    )


def load_json_file(file_path: Path) -> List[Dict]:
    """Load JSON file and convert to list of dictionaries for training.

    Supports various JSON structures:
    - Array of objects: [{"key": "value"}, ...]
    - Single object with nested data
    - OpenTelemetry-style telemetry data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []

    if isinstance(data, list):
        # Array of objects
        for item in data:
            if isinstance(item, dict):
                rows.append(item)
            else:
                rows.append({"content": str(item), "source": file_path.name})
    elif isinstance(data, dict):
        # Check for common nested structures
        # Deployment-style: {"use_cases": [...]} or generic {"data": [...], "items": [...]}
        nested_keys = ['use_cases', 'deployments', 'records', 'data', 'items', 'entries']

        found_nested = False
        for key in nested_keys:
            if key in data and isinstance(data[key], list):
                found_nested = True
                for item in data[key]:
                    if isinstance(item, dict):
                        item['_category'] = key
                        item['_source'] = file_path.name
                        rows.append(item)
                    else:
                        rows.append({
                            "content": str(item),
                            "category": key,
                            "source": file_path.name
                        })

        # If no nested arrays found, treat as single record
        if not found_nested:
            # Flatten nested dictionaries for training
            flat_row = flatten_dict(data)
            flat_row['_source'] = file_path.name
            rows.append(flat_row)

    return rows


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary for easier text conversion."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert list to string representation
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def load_txt_file(file_path: Path, chunk_size: int = 1000) -> List[Dict]:
    """Load TXT file and chunk into training samples.

    Splits text by paragraphs or fixed chunk size for training.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    rows = []

    # First try splitting by sections (double newlines or markdown headers)
    sections = []

    # Split by markdown-style headers
    import re
    header_pattern = r'\n(?=(?:SECTION|#{1,3}|[A-Z][A-Z\s]+:))'
    parts = re.split(header_pattern, content)

    for part in parts:
        part = part.strip()
        if len(part) > 50:  # Minimum meaningful content
            sections.append(part)

    # If no good sections found, split by paragraphs
    if len(sections) <= 1:
        paragraphs = content.split('\n\n')
        sections = [p.strip() for p in paragraphs if len(p.strip()) > 50]

    # If still too few sections, chunk by size
    if len(sections) <= 1 and len(content) > chunk_size:
        sections = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            if chunk.strip():
                sections.append(chunk.strip())

    for i, section in enumerate(sections):
        rows.append({
            "content": section,
            "source": file_path.name,
            "section_index": i + 1,
            "total_sections": len(sections)
        })

    return rows


def json_row_to_text(row: Dict) -> str:
    """Convert a JSON record into training text.

    Handles OpenTelemetry-style data with special formatting.
    """
    category = row.pop('_category', None)
    source = row.pop('_source', None)

    parts: List[str] = []

    # Add context prefix for provisioning/deployment
    if category:
        category_prompts = {
            'use_cases': "Provisioning use case. Deploy or create the following:",
            'deployments': "Deployment configuration. Provision the following:",
            'data': "Cloud provisioning data:",
            'items': "Provisioning item:",
            'records': "Deployment record:",
        }
        prefix = category_prompts.get(category, f"Cloud provisioning - {category}:")
    else:
        prefix = "You are an AI assistant for cloud provisioning and deployment. Use the following:"

    for k, v in row.items():
        if v is None:
            continue
        sv = str(v)
        if sv.strip() == "" or sv.strip().lower() == "nan":
            continue
        parts.append(f"{k}: {sv}")

    return f"{prefix}\n\n" + " | ".join(parts) + "\n\nSummary:"


def txt_row_to_text(row: Dict) -> str:
    """Convert a TXT chunk into training text."""
    content = row.get('content', '')
    source = row.get('source', 'unknown')

    return (
        "You are an AI assistant for cloud operations and provisioning. "
        f"Based on the following knowledge from {source}, provide expert guidance:\n\n"
        f"{content}\n\n"
        "Summary and key insights:"
    )


class MultiFormatCausalLMDataset(Dataset):
    """Dataset that handles multiple file formats: CSV, JSON, TXT, PDF."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        source_type: str = 'csv',  # 'csv', 'json', 'txt', 'pdf'
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_type = source_type

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx].to_dict()

        # Choose text conversion based on source type
        if self.source_type == 'json':
            text = json_row_to_text(row.copy())
        elif self.source_type == 'txt':
            text = txt_row_to_text(row)
        else:  # csv, pdf, or default
            text = row_to_text(row)

        # Tokenize a single training sample. We do NOT pad here; padding is handled
        # by the DataCollator dynamically per batch.
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        # Trainer expects plain tensors (not 1xT)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item


# Alias for backwards compatibility
CSVCausalLMDataset = MultiFormatCausalLMDataset


def keep_only_required_model_files(model_dir: Path) -> None:
    """Ensure the output folder contains exactly:

    /models
      config.json
      tokenizer.json
      pytorch_model.bin

    HuggingFace sometimes writes extra files (generation_config.json, tokenizer_config.json, etc.).
    This function removes extras after saving.

    NOTE: Using a fast tokenizer, `tokenizer.json` is self-contained.
    """

    required = {"config.json", "tokenizer.json", "pytorch_model.bin"}

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


def train(cfg: TrainConfig) -> None:
    # Set deterministic seeds for reproducibility.
    set_seed(cfg.seed)

    # Load dataset from multiple file formats
    # Supported: CSV, JSON, TXT, PDF
    print("=" * 80)
    print("🚀 VaLLM Training")
    print(f"📁 Supported formats: {', '.join(cfg.file_types)}")
    print("=" * 80)

    if cfg.dataset_dir is not None:
        if not cfg.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {cfg.dataset_dir}")

        frames = []
        total_files = 0

        # Skip non-provisioning datasets (observability, troubleshooting)
        def _skip_non_provisioning(path: Path) -> bool:
            name = path.name.lower()
            return "observability" in name or "troubleshoot" in name or "incident" in name

        # Process CSV files
        if 'csv' in cfg.file_types:
            csv_paths = sorted(p for p in cfg.dataset_dir.glob("*.csv") if p.is_file() and not _skip_non_provisioning(p))
            if csv_paths:
                print(f"📊 Found {len(csv_paths)} CSV files...")
                primary = cfg.dataset_dir / cfg.primary_csv
                if primary.exists():
                    csv_paths = [primary] + [p for p in csv_paths if p.resolve() != primary.resolve()]

                for p in csv_paths:
                    try:
                        frame = pd.read_csv(p, on_bad_lines='warn')
                        frame['_source_type'] = 'csv'
                        frames.append(frame)
                        total_files += 1
                        print(f"    ✓ Loaded {len(frame)} rows from {p.name}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not parse CSV {p}: {e}")

        # Process JSON files
        if 'json' in cfg.file_types:
            json_paths = sorted(p for p in cfg.dataset_dir.glob("*.json") if p.is_file() and not _skip_non_provisioning(p))
            if json_paths:
                print(f"📋 Found {len(json_paths)} JSON files...")
                for p in json_paths:
                    try:
                        rows = load_json_file(p)
                        if rows:
                            frame = pd.DataFrame(rows)
                            frame['_source_type'] = 'json'
                            frames.append(frame)
                            total_files += 1
                            print(f"    ✓ Loaded {len(rows)} records from {p.name}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not parse JSON {p}: {e}")

        # Process TXT files
        if 'txt' in cfg.file_types:
            txt_paths = sorted(p for p in cfg.dataset_dir.glob("*.txt") if p.is_file() and not _skip_non_provisioning(p))
            if txt_paths:
                print(f"📝 Found {len(txt_paths)} TXT files...")
                for p in txt_paths:
                    try:
                        rows = load_txt_file(p)
                        if rows:
                            frame = pd.DataFrame(rows)
                            frame['_source_type'] = 'txt'
                            frames.append(frame)
                            total_files += 1
                            print(f"    ✓ Loaded {len(rows)} chunks from {p.name}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not parse TXT {p}: {e}")

        # Process PDF files
        if 'pdf' in cfg.file_types:
            pdf_paths = sorted(p for p in cfg.dataset_dir.glob("*.pdf") if p.is_file() and not _skip_non_provisioning(p))
            if pdf_paths:
                if HAVE_PDF:
                    print(f"📄 Found {len(pdf_paths)} PDF files...")
                    for p in pdf_paths:
                        try:
                            reader = PyPDF2.PdfReader(p)
                            rows = []
                            for page_num, page in enumerate(reader.pages):
                                text = page.extract_text()
                                if text and text.strip():
                                    rows.append({
                                        "content": text.strip(),
                                        "source": p.name,
                                        "page": page_num + 1
                                    })
                            if rows:
                                frame = pd.DataFrame(rows)
                                frame['_source_type'] = 'pdf'
                                frames.append(frame)
                                total_files += 1
                                print(f"    ✓ Loaded {len(rows)} pages from {p.name}")
                        except Exception as e:
                            print(f"⚠️  Warning: Could not read PDF {p}: {e}")
                else:
                    print(f"⚠️  Warning: Found {len(pdf_paths)} PDF files but PyPDF2 is not installed. Skipping.")

        if not frames:
            raise FileNotFoundError(f"No valid data files found in: {cfg.dataset_dir}")

        print(f"\n✅ Loaded {total_files} files total")
        df = pd.concat(frames, ignore_index=True)
        dataset_label = str(cfg.dataset_dir)
    else:
        # Single file mode
        if not cfg.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {cfg.dataset_path}")

        suffix = cfg.dataset_path.suffix.lower()
        print("🧾 Single-file mode")
        print(f"   File: {cfg.dataset_path.name}")
        print(f"   Type: {suffix.lstrip('.')}")
        if suffix == '.csv':
            df = pd.read_csv(cfg.dataset_path, on_bad_lines='warn')
            df['_source_type'] = 'csv'
        elif suffix == '.json':
            rows = load_json_file(cfg.dataset_path)
            df = pd.DataFrame(rows)
            df['_source_type'] = 'json'
        elif suffix == '.txt':
            rows = load_txt_file(cfg.dataset_path)
            df = pd.DataFrame(rows)
            df['_source_type'] = 'txt'
        elif suffix == '.pdf':
            if not HAVE_PDF:
                raise ImportError("PyPDF2 is required for PDF files. Install with: pip install PyPDF2")
            reader = PyPDF2.PdfReader(cfg.dataset_path)
            rows = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    rows.append({
                        "content": text.strip(),
                        "source": cfg.dataset_path.name,
                        "page": page_num + 1
                    })
            df = pd.DataFrame(rows)
            df['_source_type'] = 'pdf'
            print(f"   Pages extracted: {len(rows)}")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        dataset_label = str(cfg.dataset_path)

    if len(df) == 0:
        raise ValueError(f"Dataset is empty: {dataset_label}")

    print(f"📈 Total training samples: {len(df)}")
    if "_source_type" in df.columns:
        counts = df["_source_type"].value_counts().to_dict()
        counts_str = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"📌 Sample distribution: {counts_str}")

    # Load a HuggingFace causal language model + tokenizer for text generation.
    # Default is Mistral 7B Instruct; swap to a smaller model (e.g. distilgpt2)
    # if you want a fast CPU-only smoke test.
    print(f"🧠 Model: {cfg.model_name_or_path}")
    print(f"📦 Output: {cfg.output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path)

    # Many GPT-like tokenizers don't define pad_token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Wrap the CSV as a Dataset that yields tokenized examples.
    dataset = CSVCausalLMDataset(df=df, tokenizer=tokenizer, max_length=cfg.text_max_length)

    # Standard causal LM collator (creates `labels` from `input_ids` for next-token prediction).
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Use GPU when available; pick bf16/fp16 automatically.
    use_cuda = torch.cuda.is_available()
    bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    fp16 = bool(use_cuda and not bf16)

    device_count = torch.cuda.device_count() if use_cuda else 1
    steps_per_epoch = math.ceil(len(dataset) / (cfg.per_device_train_batch_size * device_count))
    effective_steps = max(1, math.ceil(steps_per_epoch / cfg.gradient_accumulation_steps))
    log_every = max(1, effective_steps // 10)

    print("⚙️  Training config")
    print(f"   Samples: {len(dataset)}")
    print(f"   Batch size/device: {cfg.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {cfg.gradient_accumulation_steps}")
    print(f"   Epochs: {cfg.num_train_epochs}")
    print(f"   Logging steps: {log_every}")
    print(f"   Device: {'cuda' if use_cuda else 'cpu'} ({device_count} device(s))")

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir / "_checkpoints"),
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
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

    class ProgressPrinter(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            epoch = logs.get("epoch")
            step = state.global_step
            parts = [f"step={step}"]
            if epoch is not None:
                parts.append(f"epoch={epoch:.2f}")
            if loss is not None:
                parts.append(f"loss={loss:.4f}")
            if lr is not None:
                parts.append(f"lr={lr:.2e}")
            print("🟢 " + " | ".join(parts))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[ProgressPrinter()],
    )

    trainer.train()

    # Save in HuggingFace format into ./model.
    # This is what produces:
    # - config.json
    # - model weights (pytorch_model.bin or model.safetensors depending on environment)
    trainer.model.save_pretrained(cfg.output_dir)
    # This writes tokenizer.json (plus possible extra tokenizer files).
    tokenizer.save_pretrained(cfg.output_dir)

    # Ensure weights file is exactly pytorch_model.bin.
    ensure_pytorch_bin_weights(cfg.output_dir)

    # Enforce output structure exactly as requested
    keep_only_required_model_files(cfg.output_dir)

    # Quick sanity check
    required_paths = [
        cfg.output_dir / "config.json",
        cfg.output_dir / "tokenizer.json",
        cfg.output_dir / "pytorch_model.bin",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"Model export incomplete; missing files: {missing}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM on multi-format data (CSV, JSON, TXT, PDF)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path("app") / "data" / "datasets" / "cloud_deployments.csv"),
        help="Path to training file (CSV, JSON, TXT, or PDF)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        nargs="?",
        const="",
        default=str(Path("app") / "data" / "datasets"),
        help=(
            "If set, train on all supported files in this folder (recommended). "
            "Pass no value to disable and train only on --dataset."
        ),
    )
    parser.add_argument(
        "--primary-csv",
        type=str,
        default="cloud_deployments.csv",
        help="CSV to prioritize first when training on --dataset-dir",
    )
    parser.add_argument(
        "--file-types",
        type=str,
        default="csv,json,txt,pdf",
        help="Comma-separated list of file types to include (default: csv,json,txt,pdf)",
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
        default=str(Path("app") / "data" / "models"),
        help="Output folder for trained model weights",
    )
    parser.add_argument("--text-max-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Parse file types
    file_types = [ft.strip().lower() for ft in args.file_types.split(',')]
    valid_types = {'csv', 'json', 'txt', 'pdf'}
    invalid = set(file_types) - valid_types
    if invalid:
        parser.error(f"Invalid file types: {invalid}. Valid types: {valid_types}")

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
        file_types=file_types,
    )


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
