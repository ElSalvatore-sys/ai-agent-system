import hashlib
import json
import os
from datetime import datetime
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import get_logger


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that records an *immutable* append-only audit trail.

    Each log entry is written as a JSON line that includes a cryptographic
    hash-chain: every record stores the SHA-256 hash of the *previous* record
    (or all zeros for the very first record). This makes tampering evident – if
    a past entry is modified or removed, downstream hashes will fail to match.
    """

    def __init__(self, app, log_path: str | os.PathLike = "logs/audit.log"):
        super().__init__(app)
        self.logger = get_logger("audit")
        self.log_path = os.fspath(log_path)
        # ensure directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        # initialise previous hash (compute from last line if file exists)
        self.prev_hash = self._compute_last_hash()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        try:
            await self._write_audit_entry(request, response)
        except Exception as exc:  # noqa: BLE001 – ensure audit errors don't break request
            self.logger.error("Failed to write audit log: %s", exc)
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_last_hash(self) -> str:
        """Return SHA-256 of the last log entry, or 64 zeros if none."""
        if not os.path.isfile(self.log_path):
            return "0" * 64
        try:
            with open(self.log_path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                pos = fh.tell() - 1
                while pos > 0 and fh.read(1) != b"\n":
                    pos -= 1
                    fh.seek(pos, os.SEEK_SET)
                line = fh.readline()
                if not line:
                    return "0" * 64
                record = json.loads(line)
                return record.get("hash", "0" * 64)
        except Exception:
            return "0" * 64

    async def _write_audit_entry(self, request: Request, response: Response):
        now = datetime.utcnow().isoformat()
        entry = {
            "timestamp": now,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "user_id": getattr(request.state, "user_id", None),
            "client_ip": request.client.host if request.client else None,
            "previous_hash": self.prev_hash,
        }
        # Compute new hash over JSON without the hash field
        hash_input = json.dumps(entry, sort_keys=True).encode()
        entry_hash = hashlib.sha256(hash_input).hexdigest()
        entry["hash"] = entry_hash
        # Append to file atomically
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        self.prev_hash = entry_hash
