import asyncio

from app.services.ai.ml.precompute import process_csv_files, row_to_text
from app.services.ai.ml.specialist_profile import filter_specialist_csv_paths


def test_row_to_text_skips_empty_values():
    text = row_to_text({"question": "What is Kubernetes?", "answer": "Container orchestration", "notes": None})

    assert "question: What is Kubernetes?" in text
    assert "answer: Container orchestration" in text
    assert "notes:" not in text


def test_filter_specialist_csv_paths_excludes_non_it_dirs(tmp_path):
    cloud = tmp_path / "cloud" / "cloud.csv"
    finance = tmp_path / "finance" / "finance.csv"
    uploaded = tmp_path / "uploaded" / "notes.csv"

    for path in (cloud, finance, uploaded):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a\nb\n", encoding="utf-8")

    filtered = filter_specialist_csv_paths([cloud, finance, uploaded], tmp_path)

    assert cloud in filtered
    assert uploaded in filtered
    assert finance not in filtered


def test_process_csv_files_only_reads_it_scoped_csvs(tmp_path):
    cloud = tmp_path / "cloud" / "cloud.csv"
    finance = tmp_path / "finance" / "finance.csv"

    cloud.parent.mkdir(parents=True)
    finance.parent.mkdir(parents=True)

    cloud.write_text("question,answer\nWhat is EKS?,Managed Kubernetes\n", encoding="utf-8")
    finance.write_text("question,answer\nWhat is ROI?,Return on investment\n", encoding="utf-8")

    texts, metadatas, content_ids = asyncio.run(process_csv_files(tmp_path, cloud))

    assert len(texts) == 1
    assert "EKS" in texts[0]
    assert len(metadatas) == 1
    assert metadatas[0]["raw"]["question"] == "What is EKS?"
    assert metadatas[0]["raw"]["_dataset_group"] == "cloud"
    assert content_ids == ["csv_0"]
