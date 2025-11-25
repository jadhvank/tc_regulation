from pathlib import Path
import pandas as pd

from src.ingestion.csv_ingestor import csv_to_chunks
from src.ingestion.fs_ingestor import folder_to_chunks


def test_csv_to_chunks(tmp_path):
	df = pd.DataFrame([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
	p = tmp_path / "t.csv"
	df.to_csv(p, index=False)
	chunks = csv_to_chunks(p)
	assert len(chunks) == 2
	assert chunks[0]["metadata"]["file"] == "t.csv"


def test_folder_to_chunks(tmp_path):
	# csv
	df = pd.DataFrame([{"c": 3}])
	p = tmp_path / "sub"
	p.mkdir()
	csvp = p / "f.csv"
	df.to_csv(csvp, index=False)
	# txt
	txtp = p / "note.txt"
	txtp.write_text("hello")
	chunks = folder_to_chunks(tmp_path)
	assert any("hello" in c["text"] for c in chunks)
	assert any(c["metadata"].get("file") == "f.csv" for c in chunks if "metadata" in c)


