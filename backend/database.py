import os
import json
import sqlite3
import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    col_count INTEGER NOT NULL,
                    is_synthetic INTEGER NOT NULL DEFAULT 0,
                    real_file_missing INTEGER NOT NULL DEFAULT 0,
                    real_file_path TEXT,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS configs (
                    dataset_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    completed INTEGER NOT NULL,
                    total INTEGER NOT NULL,
                    current_step TEXT,
                    elapsed_ms REAL,
                    error TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_results (
                    dataset_id TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    summary_json TEXT,
                    features_json TEXT,
                    experiments_json TEXT,
                    personas_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, config_hash)
                )
            """)
            conn.commit()

    # --- Datasets ---

    def save_dataset(
        self,
        dataset_id: str,
        name: str,
        row_count: int,
        col_count: int,
        is_synthetic: bool,
        path: str,
        created_at: str,
        real_file_missing: bool = False,
        real_file_path: Optional[str] = None,
    ):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO datasets 
                (id, name, row_count, col_count, is_synthetic, real_file_missing, real_file_path, path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    name,
                    row_count,
                    col_count,
                    1 if is_synthetic else 0,
                    1 if real_file_missing else 0,
                    real_file_path,
                    path,
                    created_at,
                ),
            )
            conn.commit()

    def get_datasets(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets ORDER BY created_at ASC")
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "row_count": row["row_count"],
                    "col_count": row["col_count"],
                    "is_synthetic": bool(row["is_synthetic"]),
                    "real_file_missing": bool(row["real_file_missing"]),
                    "real_file_path": row["real_file_path"],
                    "path": row["path"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def delete_dataset(self, dataset_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            cursor.execute("DELETE FROM configs WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM jobs WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM pipeline_results WHERE dataset_id = ?", (dataset_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    # --- Configs ---

    def save_config(self, dataset_id: str, config_dict: Dict[str, Any], config_hash: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO configs (dataset_id, config_json, config_hash, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (dataset_id, json.dumps(config_dict), config_hash, datetime.datetime.now().isoformat()),
            )
            conn.commit()

    def get_config(self, dataset_id: str) -> Optional[Tuple[Dict[str, Any], str]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT config_json, config_hash FROM configs WHERE dataset_id = ?", (dataset_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["config_json"]), row["config_hash"]
            return None

    # --- Pipeline Results ---

    def save_pipeline_cache(
        self,
        dataset_id: str,
        config_hash: str,
        summary: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
        experiments: Optional[Dict[str, Any]] = None,
        personas: Optional[Dict[str, Any]] = None,
    ):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT summary_json, features_json, experiments_json, personas_json FROM pipeline_results WHERE dataset_id = ? AND config_hash = ?",
                (dataset_id, config_hash),
            )
            existing = cursor.fetchone()

            def _serializer(o):
                if hasattr(o, "model_dump"):
                    return o.model_dump()
                if hasattr(o, "dict"):
                    return o.dict()
                if isinstance(o, (np.integer, np.int64, np.int32)):
                    return int(o)
                if isinstance(o, (np.floating, np.float64, np.float32)):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return str(o)

            s_json = json.dumps(summary, default=_serializer) if summary is not None else (existing["summary_json"] if existing else None)
            f_json = json.dumps(features, default=_serializer) if features is not None else (existing["features_json"] if existing else None)
            e_json = json.dumps(experiments, default=_serializer) if experiments is not None else (existing["experiments_json"] if existing else None)
            p_json = json.dumps(personas, default=_serializer) if personas is not None else (existing["personas_json"] if existing else None)

            cursor.execute(
                """
                INSERT OR REPLACE INTO pipeline_results
                (dataset_id, config_hash, summary_json, features_json, experiments_json, personas_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    config_hash,
                    s_json,
                    f_json,
                    e_json,
                    p_json,
                    datetime.datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def get_pipeline_cache(self, dataset_id: str, config_hash: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT summary_json, features_json, experiments_json, personas_json FROM pipeline_results WHERE dataset_id = ? AND config_hash = ?",
                (dataset_id, config_hash),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "summary": json.loads(row["summary_json"]) if row["summary_json"] else None,
                    "features": json.loads(row["features_json"]) if row["features_json"] else None,
                    "experiments": json.loads(row["experiments_json"]) if row["experiments_json"] else None,
                    "personas": json.loads(row["personas_json"]) if row["personas_json"] else None,
                }
            return None

    # --- Jobs ---

    def save_job(self, job_dict: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO jobs
                (job_id, dataset_id, status, completed, total, current_step, elapsed_ms, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_dict["job_id"],
                    job_dict["dataset_id"],
                    job_dict["status"],
                    job_dict.get("completed", 0),
                    job_dict.get("total", 16),
                    job_dict.get("current_step", ""),
                    job_dict.get("elapsed_ms", 0.0),
                    job_dict.get("error"),
                    job_dict.get("created_at", datetime.datetime.now().isoformat()),
                ),
            )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
