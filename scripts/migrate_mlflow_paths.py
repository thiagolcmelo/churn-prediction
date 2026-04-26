"""
One-time migration: rewrite host absolute paths in mlflow.db to the container path.

Run once BEFORE `docker compose up`:
    python scripts/migrate_mlflow_paths.py

Safe to re-run — only rows that still contain the old prefix are updated.
"""

import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "mlflow.db"
CONTAINER_ROOT = "/mlflow/mlruns"


def _replace(uri: str) -> str:
    return re.sub(r"^.*/mlruns/", f"{CONTAINER_ROOT}/", uri)


def migrate(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        exp_rows = conn.execute(
            "SELECT experiment_id, artifact_location FROM experiments "
            "WHERE artifact_location NOT LIKE '/mlflow/mlruns/%'"
        ).fetchall()
        for exp_id, loc in exp_rows:
            conn.execute(
                "UPDATE experiments SET artifact_location=? WHERE experiment_id=?",
                (_replace(loc), exp_id),
            )

        run_rows = conn.execute(
            "SELECT run_uuid, artifact_uri FROM runs "
            "WHERE artifact_uri NOT LIKE '/mlflow/mlruns/%'"
        ).fetchall()
        for run_uuid, uri in run_rows:
            conn.execute(
                "UPDATE runs SET artifact_uri=? WHERE run_uuid=?",
                (_replace(uri), run_uuid),
            )

        conn.commit()
        print(f"Migrated {len(exp_rows)} experiment(s) and {len(run_rows)} run(s).")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
