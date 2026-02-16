"""
VaLLM Precompute Script - Build FAISS Index from CSV and JSON Data

This script reads all CSVs and JSONs from app/data/datasets/*.csv and *.json,
generates embeddings using sentence-transformers, and saves a FAISS index to
app/data/vectorstore/. JSON files like deployments.json (use_cases) are included
for training, matching train.py behaviour.

QUICK START (run from va_llm_v1 root directory):
================================================

    # Build the FAISS index
    python ./app/precompute.py

    # Then train the LLM (optional)
    python ./app/train.py --num-train-epochs 1

    # Start the FastAPI server
    python -m app.app

INPUT (datasets):  app/data/datasets/  (*.csv and *.json)
OUTPUT:
=======
    app/data/vectorstore/
        faiss_index.bin    - FAISS vector index (IndexFlatIP, cosine similarity)
        documents.pkl      - Original documents + metadata
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ============================================================================
# EMBEDDING MODEL - Must match embeddings.py at runtime
# Use a sentence-transformers model (e.g. all-MiniLM-L6-v2). Do NOT use
# causal LLMs like Qwen/Qwen2.5-3B; they are not built for similarity embeddings.
# ============================================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING_MODEL_NAME = "roneneldan/TinyStories-1M"          # ~4MB, 1M params

# ============================================================================


def row_to_text(row: Dict[str, Any]) -> str:
    """
    Convert one CSV row into text for embedding.

    For cloud_deployments rows (has 'prompt' + 'intent'):
      Use the prompt as primary text plus intent and key discriminating fields.
      This keeps embeddings focused so "deploy ec2 instance" matches provision_vm
      training data, not random provision_docker rows.

    For other rows (non-deployment CSVs):
      Use description/category or concatenate non-empty fields.

    The FULL row is always stored in metadata.raw for payload generation.
    """
    prompt = _clean(row.get("prompt"))
    intent = _clean(row.get("intent"))

    # Deployment rows: prompt-focused embedding
    if prompt and intent:
        parts = [prompt, f"intent: {intent}"]
        # Add key fields per intent type to sharpen discrimination
        for field in ("instance_type", "cloud_provider", "region", "os",
                      "cluster_name", "node_count", "node_type", "kubernetes_version",
                      "docker_image", "container_name", "ports",
                      "database_engine", "database_name",
                      "app_name", "app_port",
                      "server_name", "http_port"):
            val = _clean(row.get(field))
            if val:
                parts.append(f"{field}: {val}")
        return " | ".join(parts)

    # Non-deployment rows: use description or full row
    desc = _clean(row.get("description"))
    category = _clean(row.get("category"))
    if desc:
        parts = []
        if category:
            parts.append(f"category: {category}")
        parts.append(desc)
        resolution = _clean(row.get("resolution_or_recommendation"))
        if resolution:
            parts.append(f"resolution: {resolution}")
        return " | ".join(parts)

    # Fallback: all non-empty fields
    parts: List[str] = []
    for k, v in row.items():
        val = _clean(v)
        if val:
            parts.append(f"{k}: {val}")
    return " | ".join(parts)


def _clean(v: Any) -> str:
    """Return cleaned string or empty string for None/NaN."""
    if v is None:
        return ""
    sv = str(v).strip()
    if sv.lower() in ("", "nan"):
        return ""
    return sv


def load_json_rows(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON file and return list of row dicts for embedding.

    Supports:
    - deployments.json: object with "use_cases" array (prompt, intent, payload, etc.)
    - Other JSON: array of objects, or dict with nested keys (use_cases, data, items, etc.)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    if isinstance(data, list):
        for item in data:
            rows.append(item if isinstance(item, dict) else {"content": str(item), "source": file_path.name})
        return rows

    if not isinstance(data, dict):
        return rows

    # deployments.json-style: { "metadata": {...}, "use_cases": [ {...}, ... ] }
    if "use_cases" in data and isinstance(data["use_cases"], list):
        for item in data["use_cases"]:
            if isinstance(item, dict):
                rows.append(item)
        return rows

    # Other nested structures (deployment/provisioning: use_cases, data, items)
    nested_keys = [
        "use_cases", "deployments", "records", "data", "items", "entries",
    ]
    for key in nested_keys:
        if key in data and isinstance(data[key], list):
            for item in data[key]:
                if isinstance(item, dict):
                    item = {**item, "_category": key, "_source": file_path.name}
                    rows.append(item)
                else:
                    rows.append({"content": str(item), "category": key, "source": file_path.name})
            return rows

    # Single object: treat as one record
    flat = {"_source": file_path.name}
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            flat[k] = str(v)
        else:
            flat[k] = v
    rows.append(flat)
    return rows


def use_case_to_row(uc: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a deployments.json use_case so row_to_text() can use it (prompt, intent, payload fields)."""
    row = {
        "prompt": uc.get("prompt"),
        "intent": uc.get("intent"),
        "deployment_id": uc.get("deployment_id"),
        "endpoint_url": uc.get("endpoint_url"),
    }
    payload = uc.get("payload") or {}
    if isinstance(payload, dict):
        row.update(payload)
    return row


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS IndexFlatIP index from normalized embeddings (cosine similarity)."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={embeddings.shape}")

    dim = embeddings.shape[1]

    # Normalize for cosine similarity via Inner Product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute sentence-transformer embeddings and build a FAISS index")

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path("app") / "data" / "datasets" / "cloud_deployments.csv"),
        help="Primary CSV dataset (used first if --dataset-dir is set)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(Path("app") / "data" / "datasets"),
        help="If set, embed all CSV and JSON files in this folder (recommended)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help=f"SentenceTransformer model name. Default: {EMBEDDING_MODEL_NAME}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("app") / "data" / "vectorstore"),
        help="Directory where faiss_index.bin and documents.pkl will be written",
    )
    parser.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    if dataset_dir is None:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    else:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # STEP 1: DISCOVER AND LOAD CSV + JSON FILES
    # ==========================================================================
    print("\n" + "=" * 70)
    print("VaLLM PRECOMPUTE - Building FAISS Vector Index")
    print("=" * 70)

    # Collect (row_dict, source_path, is_json_use_case) for unified processing
    all_rows: List[tuple] = []  # (row_dict, source_str, is_json_use_case)
    csv_file_count = 0
    json_file_count = 0

    if dataset_dir is not None:
        def _skip_non_provisioning(path: Path) -> bool:
            name = path.name.lower()
            return "observability" in name or "troubleshoot" in name or "incident" in name
        csv_paths = sorted(p for p in dataset_dir.glob("*.csv") if p.is_file() and not _skip_non_provisioning(p))
        json_paths = sorted(p for p in dataset_dir.glob("*.json") if p.is_file() and not _skip_non_provisioning(p))

        if not csv_paths and not json_paths:
            raise FileNotFoundError(f"No CSV or JSON files found in: {dataset_dir}")

        if dataset_path.exists():
            if dataset_path.suffix.lower() == ".csv" and dataset_path not in csv_paths:
                csv_paths = [dataset_path] + list(csv_paths)
            elif dataset_path.suffix.lower() == ".json" and dataset_path not in json_paths:
                json_paths = [dataset_path] + list(json_paths)

        print(f"\n[DIR] Found {len(csv_paths)} CSV and {len(json_paths)} JSON file(s) in {dataset_dir}:")
        print("-" * 50)

        for idx, p in enumerate(csv_paths, 1):
            try:
                print(f"\n[{idx}/{len(csv_paths)}] CSV: {p.name}")
                frame = pd.read_csv(p, on_bad_lines='warn')
                row_count = len(frame)
                csv_file_count += 1
                print(f"    [OK] Loaded {row_count} rows x {len(frame.columns)} columns")
                for i, row in frame.iterrows():
                    row_dict = row.to_dict()
                    all_rows.append((row_dict, str(p), False))
            except Exception as e:
                print(f"    [WARN] Could not parse: {e}")

        for idx, p in enumerate(json_paths, 1):
            try:
                print(f"\n[{len(csv_paths) + idx}/{len(csv_paths) + len(json_paths)}] JSON: {p.name}")
                rows = load_json_rows(p)
                json_file_count += 1
                print(f"    [OK] Loaded {len(rows)} record(s)")
                # deployments.json use_cases have prompt + intent; other JSON may not
                use_cases_format = len(rows) > 0 and isinstance(rows[0], dict) and "prompt" in rows[0] and "intent" in rows[0]
                for row in rows:
                    all_rows.append((row, str(p), bool(use_cases_format and isinstance(row, dict) and "prompt" in row)))
            except Exception as e:
                print(f"    [WARN] Could not parse: {e}")

        total_rows = len(all_rows)
        print(f"\n{'='*50}")
        print(f"TOTAL: {total_rows} rows from {csv_file_count} CSV + {json_file_count} JSON file(s)")
        print(f"{'='*50}")
    else:
        # Single file (CSV or JSON)
        print(f"\n[FILE] Processing single file: {dataset_path}")
        if dataset_path.suffix.lower() == ".json":
            rows = load_json_rows(dataset_path)
            use_cases_format = len(rows) > 0 and isinstance(rows[0], dict) and "prompt" in rows[0] and "intent" in rows[0]
            for row in rows:
                all_rows.append((row, str(dataset_path), bool(use_cases_format and isinstance(row, dict) and "prompt" in row)))
            json_file_count = 1
            print(f"    [OK] Loaded {len(rows)} record(s) from JSON")
        else:
            df = pd.read_csv(dataset_path, on_bad_lines='warn')
            print(f"    [OK] Loaded {len(df)} rows x {len(df.columns)} columns")
            for i, row in df.iterrows():
                all_rows.append((row.to_dict(), str(dataset_path), False))
            csv_file_count = 1

    if not all_rows:
        raise ValueError("No rows to embed (dataset is empty or all files failed to parse)")

    # ==========================================================================
    # STEP 2: CONVERT ROWS TO TEXT
    # ==========================================================================
    print(f"\n[STEP 2] Converting rows to text documents...")
    print("-" * 50)

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    skipped = 0

    for i, (row_dict, source_path, is_json_use_case) in enumerate(all_rows):
        if is_json_use_case:
            row_for_text = use_case_to_row(row_dict)
            raw_for_meta = row_dict  # full use_case for provision-intent
        else:
            row_for_text = row_dict  # CSV row; row_to_text uses _clean() which handles NaN/None
            raw_for_meta = {str(k): ("" if pd.isna(v) else str(v)) for k, v in row_dict.items()}

        text = row_to_text(row_for_text)
        if not text.strip():
            skipped += 1
            continue

        texts.append(text)
        doc_type = "deployment" if raw_for_meta.get("intent") else None
        meta_entry = {
            "id": i,
            "text": text,
            "source": source_path,
            "raw": raw_for_meta,
        }
        if doc_type:
            meta_entry["type"] = doc_type
        metadatas.append(meta_entry)

        if (i + 1) % 50 == 0:
            print(f"    [OK] Processed {i + 1} rows...")

    print(f"\n    [OK] Converted {len(texts)} rows to text documents")
    if skipped > 0:
        print(f"    [WARN] Skipped {skipped} empty rows")

    # ==========================================================================
    # STEP 3: LOAD EMBEDDING MODEL
    # ==========================================================================
    print(f"\n[STEP 3] Loading embedding model...")
    print("-" * 50)
    print(f"    Model: {args.embedding_model}")

    model = SentenceTransformer(args.embedding_model, trust_remote_code=True)

    # Ensure tokenizer has a pad token (required for batched encode; many causal LMs don't set it)
    if getattr(model.tokenizer, "pad_token", None) is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    # Check device
    device = "cuda" if model.device.type == "cuda" else "cpu"
    print(f"    Device: {device.upper()}")
    print(f"    [OK] Model loaded successfully!")

    # ==========================================================================
    # STEP 4: GENERATE EMBEDDINGS
    # ==========================================================================
    print(f"\n[STEP 4] Generating embeddings for {len(texts)} documents...")
    print("-" * 50)
    print(f"    Batch size: {args.batch_size}")
    print(f"    Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"\n    Encoding documents (this may take a moment)...")

    import time
    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    elapsed = time.time() - start_time
    print(f"\n    [OK] Generated {len(embeddings)} embeddings in {elapsed:.2f} seconds")
    print(f"    Embedding shape: {embeddings.shape}")

    # ==========================================================================
    # STEP 5: BUILD FAISS INDEX
    # ==========================================================================
    print(f"\n[STEP 5] Building FAISS index...")
    print("-" * 50)

    index = build_faiss_index(embeddings)

    print(f"    [OK] FAISS index built with {index.ntotal} vectors")
    print(f"    Vector dimension: {index.d}")

    # ==========================================================================
    # STEP 6: SAVE TO DISK
    # ==========================================================================
    print(f"\n[STEP 6] Saving to disk...")
    print("-" * 50)

    index_path = out_dir / "faiss_index.bin"
    documents_path = out_dir / "documents.pkl"

    faiss.write_index(index, str(index_path))
    print(f"    [OK] FAISS index saved: {index_path}")

    with documents_path.open("wb") as f:
        pickle.dump(
            {
                "documents": texts,
                "metadata": metadatas,
                "content_ids": [str(m.get("id")) for m in metadatas],
            },
            f,
        )
    print(f"    [OK] Documents saved: {documents_path}")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print(f"\n{'='*70}")
    print("PRECOMPUTE COMPLETE!")
    print("=" * 70)
    print(f"""
    Summary:
    -----------------------------------------
    CSV files processed  : {csv_file_count}
    JSON files processed : {json_file_count}
    Documents indexed    : {len(texts)}
    Vectors created      : {index.ntotal}
    Vector dimension     : {index.d}
    Total time           : {elapsed:.2f} seconds

    Output files:
    -----------------------------------------
    FAISS Index  : {index_path}
    Documents    : {documents_path}

    [OK] Ready! Run 'python -m app.app' to start the server.
    """)


if __name__ == "__main__":
    main()
