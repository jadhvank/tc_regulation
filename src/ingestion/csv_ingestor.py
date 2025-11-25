from typing import List, Dict, Any
import pandas as pd
from pathlib import Path


def csv_to_chunks(file_path: str | Path, max_chars_per_chunk: int = 2000) -> List[Dict[str, Any]]:
	"""
	Read a CSV file and turn each row into a text chunk with metadata.
	"""
	file_path = Path(file_path)
	df = pd.read_csv(file_path)
	chunks: List[Dict[str, Any]] = []
	for i, row in df.iterrows():
		row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
		# simple splitting for long rows
		parts = [row_text[j : j + max_chars_per_chunk] for j in range(0, len(row_text), max_chars_per_chunk)]
		for p_idx, part in enumerate(parts):
			chunks.append(
				{
					"text": str(part),
					"metadata": {"file": str(file_path.name), "row_index": int(i), "part": int(p_idx)},
				}
			)
	return chunks


