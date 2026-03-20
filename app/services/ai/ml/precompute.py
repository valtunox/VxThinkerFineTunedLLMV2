"""
VaLLM Specialist Model - Precompute Script.

Author: Joel Otepa Wembo
https://joelwembo.com

Builds the FAISS vector index using the enhanced EmbeddingService. Processes CSV
files and documents; writes index.faiss and documents.pkl to
app/data/vectorstore/. Location: app/services/ai/ml/precompute.py.

By default the script indexes:
  - All CSV files in app/data/datasets/ (primary: documents.csv)
  - All documents in app/data/datasets/financial_documents/  (TXT, PDF, DOCX)
  - All documents in app/data/datasets/business_documents/   (PDF, DOCX, TXT)

QUICK START (run from project root directory):
================================================

    # Build the FAISS index (CSVs + documents)
    python -m app.services.ai.ml.precompute

    # Build from CSVs only (skip documents)
    python -m app.services.ai.ml.precompute --no-documents

    # Then train the LLM (optional)
    python -m app.services.ai.ml.train --num-train-epochs 1

    # Start the FastAPI server
    python -m app.app

OUTPUT:
=======
    app/data/vectorstore/
        index.faiss       - FAISS vector index
        documents.pkl      - Original documents + metadata
"""

import argparse
import asyncio
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
import pandas as pd
import sys
import io
import os

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# When run as script (e.g. python -m app.services.ai.ml.precompute), add project root AND app dir
if __name__ == "__main__":
    # precompute.py is in app/services/ai/ml/ -> parents[3] = app, parents[4] = project root
    _project_root = Path(__file__).resolve().parents[4]
    _app_dir_path = Path(__file__).resolve().parents[3]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    if str(_app_dir_path) not in sys.path:
        sys.path.insert(1, str(_app_dir_path))

# Import embedding service: works when loaded from app (services.*) or run as script (app.services.*)
try:
    from services.ai.ml.embedding import embedding_service
except ImportError:
    from app.services.ai.ml.embedding import embedding_service


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


def row_to_text(row: Dict[str, Any]) -> str:
    """Convert one CSV row into a single textual document"""
    parts: List[str] = []
    for k, v in row.items():
        if v is None:
            continue
        sv = str(v)
        if sv.strip() == "" or sv.strip().lower() == "nan":
            continue
        parts.append(f"{k}: {sv}")
    return " | ".join(parts)


async def process_csv_files(dataset_dir: Path, dataset_path: Path) -> tuple[List[str], List[Dict], List[str]]:
    """Load and process CSV files"""
    step_start = time.time()

    print("\n" + "-" * 70)
    print("📁 STEP 1: Loading CSV datasets")
    print("-" * 70)

    if dataset_dir and dataset_dir.exists():
        # Recursively find CSVs in all subdirectories (industry folders)
        csv_paths = sorted(p for p in dataset_dir.rglob("*.csv") if p.is_file())
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in: {dataset_dir} (searched recursively)")

        if dataset_path.exists():
            csv_paths = [dataset_path] + [p for p in csv_paths if p.resolve() != dataset_path.resolve()]

        print(f"  📂 Directory : {dataset_dir}")
        print(f"  📊 CSV files : {len(csv_paths)}")
        print()

        frames = []
        total_rows = 0
        for idx, p in enumerate(csv_paths, 1):
            try:
                file_start = time.time()
                frame = pd.read_csv(p, on_bad_lines='warn')
                row_count = len(frame)
                col_count = len(frame.columns)
                total_rows += row_count
                file_elapsed = time.time() - file_start
                print(f"  [{idx}/{len(csv_paths)}] ✓ {p.name}")
                print(f"           {row_count:,} rows x {col_count} columns ({_fmt_duration(file_elapsed)})")
                print(f"           Columns: {', '.join(frame.columns[:5])}{'...' if col_count > 5 else ''}")
                frames.append(frame)
            except Exception as e:
                print(f"  [{idx}/{len(csv_paths)}] ⚠️  {p.name}: {e}")
                continue

        if not frames:
            raise FileNotFoundError(f"No valid CSV files could be parsed in: {dataset_dir}")

        df = pd.concat(frames, ignore_index=True)
        print(f"\n  📊 CSV subtotal: {total_rows:,} rows from {len(frames)} file(s)")
    else:
        print(f"  📄 Single file: {dataset_path}")
        df = pd.read_csv(dataset_path, on_bad_lines='warn')
        print(f"  ✓ Loaded {len(df):,} rows x {len(df.columns)} columns")

    if len(df) == 0:
        raise ValueError("Dataset is empty")

    csv_elapsed = time.time() - step_start
    print(f"  ⏱️  CSV loading completed in {_fmt_duration(csv_elapsed)}")

    # Convert rows to text
    conv_start = time.time()
    print(f"\n" + "-" * 70)
    print(f"📝 STEP 2: Converting {len(df):,} rows to text documents")
    print("-" * 70)

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    content_ids: List[str] = []
    skipped = 0
    total = len(df)
    report_interval = max(1, total // 10)

    for i, row in df.iterrows():
        row_dict = row.to_dict()
        text = row_to_text(row_dict)
        if not text.strip():
            skipped += 1
            continue

        texts.append(text)
        content_ids.append(f"csv_{i}")
        metadatas.append({
            "id": int(i),
            "text": text,
            "source": "csv",
            "raw": {str(k): ("" if pd.isna(v) else str(v)) for k, v in row_dict.items()},
        })

        processed = int(i) + 1
        if processed % report_interval == 0 or processed == total:
            pct = processed / total * 100
            print(f"  ✓ {processed:,}/{total:,} rows ({pct:.0f}%)", flush=True)

    conv_elapsed = time.time() - conv_start
    print(f"\n  ✅ Converted {len(texts):,} rows to text documents in {_fmt_duration(conv_elapsed)}")
    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped:,} empty rows")

    return texts, metadatas, content_ids


async def process_documents(data_dir: Path) -> tuple[List[str], List[Dict], List[str]]:
    """Process PDF, DOCX, TXT files from data directories.

    Always scans financial_documents/ and business_documents/ (core data).
    Additional directories (policies, templates, training_materials) are
    included when they exist.
    """
    step_start = time.time()
    print(f"\n" + "-" * 70)
    print(f"📁 STEP 3: Processing documents (financial documents, business documents, etc.)")
    print("-" * 70)

    core_dirs = {
        'financial_documents': data_dir / 'financial_documents',
        'business_documents': data_dir / 'business_documents',
    }
    optional_dirs = {
        'policies': data_dir / 'policies',
        'templates': data_dir / 'templates',
        'training_materials': data_dir / 'training_materials',
    }
    doc_dirs = {**core_dirs, **optional_dirs}

    texts = []
    metadatas = []
    content_ids = []
    dir_stats: Dict[str, Dict[str, int]] = {}

    for dir_name, dir_path in doc_dirs.items():
        if not dir_path.exists():
            if dir_name in core_dirs:
                print(f"\n  ⚠️  Core directory missing: {dir_path}")
            continue

        dir_start = time.time()
        print(f"\n  📂 Processing {dir_name}/")
        doc_files = []
        for ext in ['.pdf', '.docx', '.doc', '.txt', '.md', '.html']:
            doc_files.extend(dir_path.rglob(f'*{ext}'))

        if not doc_files:
            print(f"     ⚠️  No documents found")
            continue

        print(f"     Found {len(doc_files)} document(s)")

        success_count = 0
        fail_count = 0
        total_chars = 0

        for idx, doc_file in enumerate(doc_files, 1):
            try:
                file_start = time.time()
                result = await embedding_service.load_and_embed_document(doc_file)
                file_elapsed = time.time() - file_start
                if result.get('success'):
                    doc_text = result['text']
                    source_label = dir_name.replace('_', ' ').title()
                    enriched_text = f"[{source_label}] {doc_file.stem}\n\n{doc_text}"
                    texts.append(enriched_text)
                    content_ids.append(f"{dir_name}_{idx}")
                    metadatas.append({
                        'source': dir_name,
                        'file_name': doc_file.name,
                        'file_path': str(doc_file),
                        'file_type': result['file_type'],
                        'text': enriched_text,
                        'text_length': result['text_length'],
                    })
                    success_count += 1
                    total_chars += result['text_length']
                    print(
                        f"     ✓ [{idx}/{len(doc_files)}] {doc_file.name} "
                        f"({result['text_length']:,} chars, {_fmt_duration(file_elapsed)})"
                    )
                else:
                    fail_count += 1
                    print(f"     ✗ [{idx}/{len(doc_files)}] {doc_file.name}: {result.get('error')}")
            except Exception as e:
                fail_count += 1
                print(f"     ✗ [{idx}/{len(doc_files)}] {doc_file.name}: {e}")

        dir_elapsed = time.time() - dir_start
        dir_stats[dir_name] = {
            "success": success_count, "failed": fail_count, "chars": total_chars,
        }
        print(
            f"     📊 {dir_name}: {success_count} loaded, {fail_count} failed, "
            f"{total_chars:,} chars total ({_fmt_duration(dir_elapsed)})"
        )

    step_elapsed = time.time() - step_start
    if texts:
        print(f"\n  ✅ Processed {len(texts)} documents total in {_fmt_duration(step_elapsed)}")
        for dn, ds in dir_stats.items():
            print(f"     {dn}: {ds['success']} docs, {ds['chars']:,} chars")
    else:
        print(f"\n  ⚠️  No documents processed")

    return texts, metadatas, content_ids


async def main_async(args):
    """Main async function"""
    global_start = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dataset_path = Path(args.dataset)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    skip_embedding_init = getattr(args, "skip_embedding_init", False)
    skip_documents = getattr(args, "no_documents", False)

    print("\n" + "=" * 80)
    print("🚀 VaLLM - PRECOMPUTE (FAISS Vector Index Builder)")
    print("=" * 80)
    print(f"  📅 Started        : {now}")
    print(f"  📂 Dataset dir    : {dataset_dir or dataset_path}")
    print(f"  📄 Documents      : {'skip' if skip_documents else 'include'}")
    print("=" * 80)

    # ── Initialize embedding service ────────────────────────────────────────
    init_start = time.time()
    if not skip_embedding_init:
        print("\n" + "-" * 70)
        print("🤖 Initializing Embedding Service")
        print("-" * 70)
    success = await embedding_service.initialize()
    if not success:
        print("  ❌ Failed to initialize embedding service")
        return

    init_elapsed = time.time() - init_start
    if not skip_embedding_init:
        print(f"  ✅ Service initialized in {_fmt_duration(init_elapsed)}")
        print(f"  📦 Model     : {embedding_service.model_name}")
        print(f"  💻 Device    : {embedding_service.device.upper()}")
        print(f"  📐 Dimension : {embedding_service.dimension}")

    # ── Process CSV files (steps 1 & 2 inside) ──────────────────────────────
    texts, metadatas, content_ids = await process_csv_files(dataset_dir, dataset_path)
    csv_count = len(texts)

    # ── Process documents ───────────────────────────────────────────────────
    doc_count = 0
    if not skip_documents:
        doc_data_dir = dataset_dir or dataset_path.parent
        doc_texts, doc_metas, doc_ids = await process_documents(doc_data_dir)
        texts.extend(doc_texts)
        metadatas.extend(doc_metas)
        content_ids.extend(doc_ids)
        doc_count = len(doc_texts)
    else:
        print(f"\n  ⏭️  Skipping document processing (--no-documents)")

    # ── Generate and store embeddings ───────────────────────────────────────
    embed_start = time.time()
    print(f"\n" + "-" * 70)
    print(f"🧠 STEP 4: Generating & storing embeddings for {len(texts):,} documents")
    print("-" * 70)
    print(f"  📊 CSV texts    : {csv_count:,}")
    print(f"  📄 Doc texts    : {doc_count:,}")
    print(f"  📝 Total        : {len(texts):,}")
    print(f"  ⏳ Embedding in progress...", flush=True)

    stored_count = await embedding_service.store_embeddings_in_faiss(
        texts=texts,
        content_type="precomputed",
        content_ids=content_ids,
        metadata_list=metadatas
    )

    embed_elapsed = time.time() - embed_start
    docs_per_sec = stored_count / embed_elapsed if embed_elapsed > 0 else 0
    print(f"\n  ✅ Stored {stored_count:,} embeddings in {_fmt_duration(embed_elapsed)}")
    print(f"     Throughput: {docs_per_sec:.1f} docs/sec")

    # ── Final stats & summary ───────────────────────────────────────────────
    stats = await embedding_service.get_faiss_stats()
    by_type = stats.get('by_type', {})
    total_elapsed = time.time() - global_start

    index_size_mb = 0.0
    meta_size_mb = 0.0
    try:
        idx_path = Path(embedding_service.index_path)
        meta_path = Path(embedding_service.metadata_path)
        if idx_path.exists():
            index_size_mb = idx_path.stat().st_size / (1024 * 1024)
        if meta_path.exists():
            meta_size_mb = meta_path.stat().st_size / (1024 * 1024)
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("🎉 PRECOMPUTE COMPLETE!")
    print("=" * 80)
    print(f"""
    📊 Summary
    {'─' * 50}
    📝 Total indexed       : {stored_count:,}
       CSV rows            : {csv_count:,}
       Documents           : {doc_count:,}
    🔢 Total vectors       : {stats['total_vectors']:,}
    📐 Vector dimension    : {stats['dimension']}
    📦 Embedding model     : {embedding_service.model_name}
    💻 Device              : {embedding_service.device.upper()}

    ⏱️  Timing
    {'─' * 50}
    🤖 Service init        : {_fmt_duration(init_elapsed)}
    🧠 Embedding + store   : {_fmt_duration(embed_elapsed)}
    ⏱️  Total wall time    : {_fmt_duration(total_elapsed)}
    🚀 Throughput          : {docs_per_sec:.1f} docs/sec

    📂 Sources breakdown
    {'─' * 50}""")
    for src_type, src_count in sorted(by_type.items()):
        print(f"       {src_type:20s} : {src_count:,}")
    print(f"""
    📂 Output files
    {'─' * 50}
    📚 FAISS Index  : {embedding_service.index_path} ({index_size_mb:.2f} MB)
    📋 Metadata     : {embedding_service.metadata_path} ({meta_size_mb:.2f} MB)

    ✅ Ready! Run 'python -m app.app' to start the server.
    """)


async def run_precompute(
    dataset_dir: Path | None = None,
    include_documents: bool = True,
    base_path: Path | None = None,
    skip_embedding_init: bool = False,
) -> bool:
    """
    Run precompute (build FAISS index from CSVs + financial/business documents).
    Callable from app startup.
    Uses default app/data/datasets. Output goes to app/data/vectorstore (via embedding service).
    Returns True on success.
    """
    _app_dir = Path(__file__).resolve().parents[3]
    base = base_path if base_path is not None else _app_dir
    data_dir = dataset_dir if dataset_dir is not None else base / "data" / "datasets"
    data_dir = Path(data_dir)
    csvs = sorted(data_dir.rglob("*.csv")) if data_dir.exists() else []
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")
    dataset_path = csvs[0]

    args = SimpleNamespace(
        dataset=str(dataset_path),
        dataset_dir=str(data_dir),
        no_documents=not include_documents,
        skip_embedding_init=skip_embedding_init,
    )
    await main_async(args)
    return True


def _app_dir() -> Path:
    """App directory (app/data/...). precompute.py is in app/services/ai/ml/."""
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute embeddings and build FAISS index (output: app/data/vectorstore)")

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(_app_dir() / "data" / "datasets" / "documents.csv"),
        help="Primary CSV dataset",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(_app_dir() / "data" / "datasets"),
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--include-documents",
        action="store_true",
        default=True,
        help="(Default: enabled) Process financial documents, business documents, and other files",
    )
    parser.add_argument(
        "--no-documents",
        action="store_true",
        default=False,
        help="Skip processing document files (CSV only)",
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

