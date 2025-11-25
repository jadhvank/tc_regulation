from typing import List, Dict, Any
from pathlib import Path
import zipfile

from src.ingestion.csv_ingestor import csv_to_chunks


def folder_to_chunks(folder_path: str | Path) -> List[Dict[str, Any]]:
	folder = Path(folder_path)
	all_chunks: List[Dict[str, Any]] = []
	for path in folder.rglob("*"):
		if path.is_file():
			if path.suffix.lower() == ".csv":
				all_chunks.extend(csv_to_chunks(path))
			elif path.suffix.lower() in {".txt", ".md"}:
				text = path.read_text(encoding="utf-8", errors="ignore")
				all_chunks.append({"text": text, "metadata": {"file": path.name}})
	return all_chunks


def unzip_to_folder(zip_file: str | Path, dest_dir: str | Path) -> Path:
	dest = Path(dest_dir)
	dest.mkdir(parents=True, exist_ok=True)
	with zipfile.ZipFile(zip_file, "r") as z:
		z.extractall(dest)
	return dest


