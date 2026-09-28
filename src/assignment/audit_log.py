"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store a redacted request preview until its output is available."""
        rid = request_id or f"{user_id or 'anonymous'}-{len(self._open) + len(self.logs) + 1}"
        self._open[rid] = {
            "request_id": rid,
            "user_id": user_id or "anonymous",
            "input": self._safe_preview(text),
            "started_at": utc_now_iso(),
            "started_monotonic": time.perf_counter(),
        }
        return rid

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Append a redacted interaction record with duration metadata."""
        rid = request_id
        if rid is None:
            candidates = [
                key for key, value in self._open.items()
                if value.get("user_id") == (user_id or "anonymous")
            ]
            rid = candidates[-1] if candidates else None

        opened = self._open.pop(rid, None) if rid else None
        started = opened.get("started_monotonic") if opened else None
        duration_ms = round((time.perf_counter() - started) * 1000, 3) if started else None
        row = {
            "request_id": rid,
            "user_id": user_id or "anonymous",
            "input": (opened or {}).get("input", ""),
            "output": self._safe_preview(text),
            "blocked": bool(blocked),
            "layer": layer,
            "timestamp": utc_now_iso(),
            "latency_ms": duration_ms,
        }
        self.logs.append(row)
        return row

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _safe_preview(text: str, limit: int = 500) -> str:
        """Return a bounded preview with common sensitive values redacted."""
        value = str(text or "")
        patterns = (
            r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{7,}\b",
            r"[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+",
            r"(?<!\d)(?:\+84|0084|0)(?:3|5|7|8|9)\d{8}(?!\d)",
            r"\b[a-z0-9.-]+\.internal(?::\d+)?\b",
            r"\b(?:password|passwd|passcode|mật\s*khẩu)\s*(?:is|=|:|là)\s*\S+",
        )
        for pattern in patterns:
            value = re.sub(pattern, "[REDACTED]", value, flags=re.IGNORECASE)
        try:
            from core.config import DEMO_SECRETS

            for secret in DEMO_SECRETS:
                if secret:
                    value = value.replace(str(secret), "[REDACTED]")
        except Exception:
            # Logging must remain best-effort and never block the request.
            pass
        return value[:limit]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
