from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
import re


def _read_csv_best_effort(file_path: Path) -> pd.DataFrame:
	"""
	Try multiple encodings commonly used in KR/legacy files to avoid UnicodeDecodeError.
	Fallback to latin1 if UTF-8 and CP949/EUC-KR fail. Skip bad lines to be robust.
	"""
	last_err: Exception | None = None
	for enc in ["utf-8", "cp949", "euc-kr", "latin1"]:
		try:
			return pd.read_csv(file_path, encoding=enc, on_bad_lines="skip")
		except Exception as e:
			last_err = e
			continue
	# If all attempts failed, re-raise the last error
	if last_err:
		raise last_err
	# Should not reach here
	return pd.read_csv(file_path)


def _read_csv_no_header_best_effort(file_path: Path, nrows: int | None = None) -> pd.DataFrame:
	"""
	Read CSV without treating the first row as header; used for header detection.
	"""
	last_err: Exception | None = None
	for enc in ["utf-8", "cp949", "euc-kr", "latin1"]:
		try:
			return pd.read_csv(file_path, encoding=enc, on_bad_lines="skip", header=None, nrows=nrows)
		except Exception as e:
			last_err = e
			continue
	if last_err:
		raise last_err
	return pd.read_csv(file_path, header=None, nrows=nrows)


def _looks_like_name(value: str) -> bool:
	if value is None:
		return False
	s = str(value).strip()
	if not s:
		return False
	# penalize pure numbers or dates
	if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
		return False
	# common header-ish pattern: letters/underscores/spaces/slashes, not too long
	return bool(re.search(r"[A-Za-z가-힣_]", s)) and len(s) <= 64


def _score_header_row(values: List[str]) -> float:
	nonempty = [str(v).strip() for v in values if str(v).strip() != ""]
	if not nonempty:
		return 0.0
	looks = sum(1 for v in nonempty if _looks_like_name(v))
	unique = len(set(nonempty))
	dup_penalty = max(0, len(nonempty) - unique)
	numeric_like = sum(1 for v in nonempty if re.fullmatch(r"[-+]?\d+(\.\d+)?", v))
	return looks * 2.0 + unique * 1.0 - dup_penalty * 0.5 - numeric_like * 1.0


def _read_csv_with_smart_header(file_path: Path, scan_rows: int = 12) -> pd.DataFrame:
	"""
	Detect header row by scanning the first few rows and choosing the one that
	looks most like column names, then return a DataFrame with proper columns set.
	"""
	df0 = _read_csv_no_header_best_effort(file_path)
	max_scan = min(scan_rows, len(df0))
	best_idx = 0
	best_score = float("-inf")
	for i in range(max_scan):
		score = _score_header_row([df0.iloc[i, j] if j < df0.shape[1] else "" for j in range(df0.shape[1])])
		if score > best_score:
			best_score = score
			best_idx = i
	# build final df: use row best_idx as header, drop rows up to that
	header_vals = [str(v).strip() if str(v).strip() != "" else f"col_{k}" for k, v in enumerate(df0.iloc[best_idx].tolist())]
	df = df0.iloc[best_idx + 1 :].copy()
	df.columns = header_vals
	df = df.reset_index(drop=True)
	return df


def csv_to_chunks(file_path: str | Path, max_chars_per_chunk: int = 2000) -> List[Dict[str, Any]]:
	"""
	Read a CSV file and turn each row into a text chunk with metadata.
	"""
	file_path = Path(file_path)
	# Use smart header detection to robustly find header row even if not the first row
	try:
		df = _read_csv_with_smart_header(file_path)
		use_smart = True
	except Exception:
		df = _read_csv_best_effort(file_path)
		use_smart = False
	chunks: List[Dict[str, Any]] = []
	for i, row in df.iterrows():
		# build structured mapping for SQL normalization
		# Use positional lookup to avoid Series ambiguity when duplicate column names exist
		structured: Dict[str, Any] = {}
		for idx, col in enumerate(list(df.columns)):
			try:
				val = row.iloc[idx]
			except Exception:
				# Fallback to label-based as a last resort
				val = row.get(col, None)
			if val is None or (pd.isna(val) if not isinstance(val, (list, tuple, dict)) else False):
				structured[str(col)] = ""
			else:
				structured[str(col)] = str(val)
		row_text = ", ".join([f"{col}: {structured[str(col)]}" for col in df.columns])
		# simple splitting for long rows
		parts = [row_text[j : j + max_chars_per_chunk] for j in range(0, len(row_text), max_chars_per_chunk)]
		for p_idx, part in enumerate(parts):
			chunks.append(
				{
					"text": str(part),
					"metadata": {"file": str(file_path.name), "row_index": int(i), "part": int(p_idx)},
					"structured": structured if p_idx == 0 else None,
				}
			)
	return chunks


