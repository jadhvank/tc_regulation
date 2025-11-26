from __future__ import annotations
from typing import Optional, Tuple
import os
import sqlite3
from pathlib import Path
from datetime import datetime

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import secrets

from src.config.settings import get_settings


def _db_conn() -> sqlite3.Connection:
	settings = get_settings()
	db_path = Path(settings.SQLITE_DB_PATH)
	conn = sqlite3.connect(str(db_path))
	conn.row_factory = sqlite3.Row
	_init_schema(conn)
	return conn


def _init_schema(conn: sqlite3.Connection) -> None:
	conn.executescript(
		"""
		CREATE TABLE IF NOT EXISTS app_secrets (
			key TEXT PRIMARY KEY,
			nonce BLOB NOT NULL,
			ciphertext BLOB NOT NULL,
			updated_at TEXT NOT NULL
		);
		"""
	)


def _get_kms_passphrase_and_salt() -> Tuple[bytes, bytes]:
	settings = get_settings()
	base_dir = Path(settings.DATA_DIR) / "secrets"
	base_dir.mkdir(parents=True, exist_ok=True)
	pass_file = base_dir / "kms_passphrase.bin"
	salt_file = base_dir / "kms_salt.bin"
	# Passphrase: env takes precedence; else stored/generated
	env_pass = os.getenv("CONFIG_KMS_PASSPHRASE")
	if env_pass:
		passphrase = env_pass.encode("utf-8")
	else:
		if pass_file.exists():
			passphrase = pass_file.read_bytes()
		else:
			passphrase = secrets.token_bytes(32)
			pass_file.write_bytes(passphrase)
	# Salt: persisted; generate if missing
	if salt_file.exists():
		salt = salt_file.read_bytes()
	else:
		salt = secrets.token_bytes(16)
		salt_file.write_bytes(salt)
	return passphrase, salt


def _derive_key() -> bytes:
	passphrase, salt = _get_kms_passphrase_and_salt()
	kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1, backend=default_backend())
	return kdf.derive(passphrase)


def set_secret(key: str, value: str) -> None:
	if not key:
		return
	plain = value.encode("utf-8")
	aes_key = _derive_key()
	aead = AESGCM(aes_key)
	nonce = secrets.token_bytes(12)
	ct = aead.encrypt(nonce, plain, associated_data=key.encode("utf-8"))
	conn = _db_conn()
	try:
		conn.execute(
			"INSERT INTO app_secrets(key, nonce, ciphertext, updated_at) VALUES (?, ?, ?, ?) "
			"ON CONFLICT(key) DO UPDATE SET nonce = excluded.nonce, ciphertext = excluded.ciphertext, updated_at = excluded.updated_at",
			(key, nonce, ct, datetime.utcnow().isoformat(timespec="seconds") + "Z"),
		)
		conn.commit()
	finally:
		conn.close()


def get_secret(key: str) -> Optional[str]:
	if not key:
		return None
	conn = _db_conn()
	try:
		row = conn.execute("SELECT nonce, ciphertext FROM app_secrets WHERE key = ?", (key,)).fetchone()
		if not row:
			return None
		nonce = row["nonce"]
		ct = row["ciphertext"]
		aes_key = _derive_key()
		aead = AESGCM(aes_key)
		pt = aead.decrypt(nonce, ct, associated_data=key.encode("utf-8"))
		return pt.decode("utf-8")
	finally:
		conn.close()


def is_set(key: str) -> bool:
	conn = _db_conn()
	try:
		row = conn.execute("SELECT 1 FROM app_secrets WHERE key = ? LIMIT 1", (key,)).fetchone()
		return row is not None
	finally:
		conn.close()



