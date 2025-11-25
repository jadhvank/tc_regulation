from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

from src.ingestion.sql_store import insert_schema_columns
from src.ingestion.csv_ingestor import _read_csv_with_smart_header


_PANDAS_TO_SIMPLE = {
	"int64": "integer",
	"Int64": "integer",
	"float64": "float",
	"boolean": "boolean",
	"bool": "boolean",
	"datetime64[ns]": "datetime",
	"object": "text",
	"string": "text",
}


def _infer_type_from_series(s: pd.Series) -> str:
	dt = str(s.dtype)
	return _PANDAS_TO_SIMPLE.get(dt, "text")


def analyze_csv_file(file_path: str | Path) -> List[Dict[str, Any]]:
	"""
	Heuristic CSV schema analysis using pandas dtypes.
	Returns: [{ name, type, position }]
	"""
	file_path = Path(file_path)
	try:
		df = _read_csv_with_smart_header(file_path)
	except Exception:
		df = pd.read_csv(file_path, on_bad_lines="skip", encoding_errors="ignore")
	schema: List[Dict[str, Any]] = []
	for idx, col in enumerate(df.columns):
		schema.append({"name": str(col), "type": _infer_type_from_series(df[col]), "position": idx})
	return schema


def analyze_and_store_schema(session_id: str, file_path: str | Path) -> List[Dict[str, Any]]:
	file_path = Path(file_path)
	cols = analyze_csv_file(file_path)
	insert_schema_columns(session_id=session_id, filename=file_path.name, columns=cols)
	return cols


