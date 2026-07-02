"""SQLite-backed feedback correction store.

C39: db_path from config, never hardcoded.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from src.api.schemas import FeedbackRequest
from src.logger import get_logger

_logger = get_logger(__name__)


class FeedbackStore:
    """SQLite-backed feedback loop + correction stats."""

    def __init__(self, db_path: str, min_corrections_for_retrain: int = 50) -> None:
        """Create or open the SQLite store at db_path; initialise schema."""
        self._db_path = db_path
        self._min_for_retrain = min_corrections_for_retrain
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create the corrections table if it does not already exist."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_prompt TEXT NOT NULL,
                    original_label TEXT NOT NULL,
                    corrected_label TEXT NOT NULL,
                    original_decision TEXT NOT NULL,
                    original_confidence REAL NOT NULL,
                    feedback_type TEXT NOT NULL
                )
                """)
            conn.commit()

    def submit_correction(self, request: FeedbackRequest) -> dict[str, Any]:
        """Insert one correction row; return stored=True + total count."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO corrections
                        (timestamp, user_prompt, original_label, corrected_label,
                         original_decision, original_confidence, feedback_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        request.user_prompt,
                        request.original_label,
                        request.corrected_label,
                        request.original_decision,
                        request.original_confidence,
                        request.feedback_type,
                    ),
                )
                conn.commit()
                total: int = conn.execute(
                    "SELECT COUNT(*) FROM corrections"
                ).fetchone()[0]
        _logger.info(
            "feedback_stored",
            extra={"total_corrections": total, "feedback_type": request.feedback_type},
        )
        return {"stored": True, "total_corrections": total}

    def get_stats(self) -> dict[str, Any]:
        """Return total corrections, breakdown by type/label, retrain_ready flag."""
        with sqlite3.connect(self._db_path) as conn:
            total: int = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]

            by_type_rows = conn.execute(
                "SELECT feedback_type, COUNT(*) FROM corrections GROUP BY feedback_type"
            ).fetchall()

            by_label_rows = conn.execute(
                "SELECT original_label, COUNT(*) FROM corrections"
                " GROUP BY original_label"
            ).fetchall()

        by_type: dict[str, int] = {row[0]: row[1] for row in by_type_rows}
        by_label: dict[str, int] = {row[0]: row[1] for row in by_label_rows}
        return {
            "total_corrections": total,
            "corrections_by_type": by_type,
            "corrections_by_original_label": by_label,
            "retrain_ready": total >= self._min_for_retrain,
        }
