import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

from src.config.settings import get_settings


class JsonFormatter(logging.Formatter):
	def format(self, record: logging.LogRecord) -> str:
		payload: Dict[str, Any] = {
			"level": record.levelname,
			"logger": record.name,
			"message": record.getMessage(),
			"time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
		}
		if record.exc_info:
			payload["exc_info"] = self.formatException(record.exc_info)
		return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger

	settings = get_settings()
	log_dir = Path(settings.LOG_DIR)
	log_dir.mkdir(parents=True, exist_ok=True)
	log_file = log_dir / "server.log"

	logger.setLevel(logging.INFO)

	# Console handler
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(JsonFormatter())
	logger.addHandler(console_handler)

	# Rotating file handler
	file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
	file_handler.setFormatter(JsonFormatter())
	logger.addHandler(file_handler)

	return logger


