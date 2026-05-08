"""
Persistence for benchmark runs and results.

Two backends are supported; selection is by file extension:
  .db  -> BenchmarkStore (SQLite)
  .csv -> CSVStore (flat CSV)

Both implement the same interface so the rest of the code is backend-agnostic.
Results are written incrementally as they complete so partial runs survive
process crashes.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.result import SimulationResult


def _make_run_id() -> str:
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    return f"{ts}_{uid}"


def _nan_none(v) -> float | None:
    if v is None:
        return None
    try:
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return v


def open_store(path: str | Path) -> "AbstractStore":
    """Return the correct store implementation based on file extension."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        return CSVStore(p)
    return BenchmarkStore(p)


class AbstractStore(ABC):
    """Common interface for benchmark result stores."""

    @abstractmethod
    def create_run(
        self,
        benchmark_name: str,
        dynamics: str,
        quick: bool,
        bench_cfg: dict | None = None,
        notes: str = "",
    ) -> str:
        """Create a run entry and return its run_id."""

    @abstractmethod
    def add_result(self, run_id: str, result: SimulationResult) -> None:
        """Persist one result immediately (incremental)."""

    @abstractmethod
    def list_runs(self, benchmark_name: str | None = None) -> pd.DataFrame:
        """Return a summary table of all runs."""

    @abstractmethod
    def list_replay_results(self) -> pd.DataFrame:
        """Return all results with associated replay NPZ files."""

    @abstractmethod
    def get_result(self, result_id: int) -> dict | None:
        """Return a single result row by id (SQLite only; None for CSV)."""

    @abstractmethod
    def export_csv(self, run_id: str, csv_path: Path) -> None:
        """Write all results for a run to a CSV file."""

    @abstractmethod
    def load_run(self, run_id: str) -> pd.DataFrame:
        """Return all results for a run as a DataFrame."""


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    benchmark       TEXT NOT NULL,
    dynamics        TEXT,
    quick           INTEGER NOT NULL DEFAULT 0,
    notes           TEXT DEFAULT '',
    config_snapshot TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    city              TEXT,
    scenario          TEXT,
    algorithm         TEXT,
    segment           INTEGER,
    nominal_speed     REAL,
    success           INTEGER,
    is_clear          INTEGER,
    crash_reason      TEXT,
    avg_solve_time    REAL,
    std_solve_time    REAL,
    min_clearance     REAL,
    ref_min_clearance REAL,
    travel_time       REAL,
    path_length       REAL,
    converge_rate     REAL,
    control_effort    REAL,
    steps             INTEGER,
    replay_npz        TEXT DEFAULT '',
    city_meta         TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_results_run  ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_algo ON results(algorithm);
CREATE INDEX IF NOT EXISTS idx_runs_bench   ON runs(benchmark);
"""

_INSERT_RUN = """
INSERT INTO runs
    (run_id, timestamp, benchmark, dynamics, quick, notes, config_snapshot)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_RESULT = """
INSERT INTO results
    (run_id, city, scenario, algorithm, segment, nominal_speed,
     success, is_clear, crash_reason,
     avg_solve_time, std_solve_time, min_clearance, ref_min_clearance,
     travel_time, path_length, converge_rate, control_effort, steps,
     replay_npz, city_meta)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


class BenchmarkStore(AbstractStore):
    """SQLite-backed store. Supports all features including replay lookup by id."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def create_run(
        self,
        benchmark_name: str,
        dynamics: str,
        quick: bool,
        bench_cfg: dict | None = None,
        notes: str = "",
    ) -> str:
        run_id    = _make_run_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        snapshot  = json.dumps(bench_cfg or {}, default=str)
        with self._conn() as conn:
            conn.execute(_INSERT_RUN,
                         (run_id, timestamp, benchmark_name, dynamics,
                          int(quick), notes, snapshot))
        return run_id

    def add_result(self, run_id: str, result: SimulationResult) -> None:
        r = result
        with self._conn() as conn:
            conn.execute(_INSERT_RESULT, (
                run_id, r.city, r.scenario, r.algorithm, r.segment, r.nominal_speed,
                int(r.success), int(r.is_clear), r.crash_reason,
                _nan_none(r.avg_solve_time), _nan_none(r.std_solve_time),
                _nan_none(r.min_clearance),  _nan_none(r.ref_min_clearance),
                _nan_none(r.travel_time),    _nan_none(r.path_length),
                _nan_none(r.converge_rate),  _nan_none(r.control_effort),
                r.steps,
                r.replay_npz,
                json.dumps(r.city_meta) if r.city_meta else "",
            ))

    def load_run(self, run_id: str) -> pd.DataFrame:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM results WHERE run_id = ?", (run_id,)
            ).fetchall()
        return _rows_to_df(rows)

    def load_latest(self, benchmark_name: str | None = None) -> tuple[str, pd.DataFrame]:
        with self._conn() as conn:
            if benchmark_name:
                row = conn.execute(
                    "SELECT run_id FROM runs WHERE benchmark = ? "
                    "ORDER BY timestamp DESC LIMIT 1",
                    (benchmark_name,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT run_id FROM runs ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
        if row is None:
            return "", pd.DataFrame()
        run_id = row["run_id"]
        return run_id, self.load_run(run_id)

    def list_runs(self, benchmark_name: str | None = None) -> pd.DataFrame:
        query  = "SELECT run_id, timestamp, benchmark, dynamics, quick, notes FROM runs"
        params: tuple = ()
        if benchmark_name:
            query  += " WHERE benchmark = ?"
            params  = (benchmark_name,)
        query += " ORDER BY timestamp DESC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def list_replay_results(self) -> pd.DataFrame:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT r.id, r.run_id, r.city, r.algorithm, r.segment, "
                "r.nominal_speed, r.success, r.replay_npz "
                "FROM results r WHERE r.replay_npz != '' "
                "ORDER BY r.run_id DESC, r.id"
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_result(self, result_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM results WHERE id = ?", (result_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete_run(self, run_id: str) -> bool:
        with self._conn() as conn:
            n = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,)).rowcount
        return n > 0

    def export_csv(self, run_id: str, csv_path: Path) -> None:
        self.load_run(run_id).to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# CSV backend
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "run_id", "timestamp", "benchmark", "dynamics", "quick",
    "city", "scenario", "algorithm", "segment", "nominal_speed",
    "success", "is_clear", "crash_reason",
    "avg_solve_time", "std_solve_time", "min_clearance", "ref_min_clearance",
    "travel_time", "path_length", "converge_rate", "control_effort", "steps",
    "replay_npz",
]


class CSVStore(AbstractStore):
    """
    Flat-CSV store. Each result row includes run metadata (run_id, benchmark,
    dynamics, quick, timestamp) so the file is self-contained.

    Limitations vs SQLite:
      - get_result(id) is not supported (returns None).
      - city_meta JSON is not stored (replay visualizer falls back to defaults).
    """

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        self._runs: dict[str, dict] = {}
        if self.csv_path.exists():
            self._load_run_index()

    def _load_run_index(self) -> None:
        try:
            df = pd.read_csv(self.csv_path, usecols=["run_id", "timestamp", "benchmark",
                                                       "dynamics", "quick"])
            for _, row in df.drop_duplicates("run_id").iterrows():
                self._runs[row["run_id"]] = dict(row)
        except Exception:
            pass

    def _write_row(self, row: dict) -> None:
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def create_run(
        self,
        benchmark_name: str,
        dynamics: str,
        quick: bool,
        bench_cfg: dict | None = None,
        notes: str = "",
    ) -> str:
        run_id    = _make_run_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        self._runs[run_id] = {
            "run_id":    run_id,
            "timestamp": timestamp,
            "benchmark": benchmark_name,
            "dynamics":  dynamics,
            "quick":     int(quick),
        }
        return run_id

    def add_result(self, run_id: str, result: SimulationResult) -> None:
        run_meta = self._runs.get(run_id, {})
        r = result
        self._write_row({
            "run_id":            run_id,
            "timestamp":         run_meta.get("timestamp", ""),
            "benchmark":         run_meta.get("benchmark", ""),
            "dynamics":          run_meta.get("dynamics", ""),
            "quick":             run_meta.get("quick", 0),
            "city":              r.city,
            "scenario":          r.scenario,
            "algorithm":         r.algorithm,
            "segment":           r.segment,
            "nominal_speed":     r.nominal_speed,
            "success":           int(r.success),
            "is_clear":          int(r.is_clear),
            "crash_reason":      r.crash_reason,
            "avg_solve_time":    _nan_none(r.avg_solve_time),
            "std_solve_time":    _nan_none(r.std_solve_time),
            "min_clearance":     _nan_none(r.min_clearance),
            "ref_min_clearance": _nan_none(r.ref_min_clearance),
            "travel_time":       _nan_none(r.travel_time),
            "path_length":       _nan_none(r.path_length),
            "converge_rate":     _nan_none(r.converge_rate),
            "control_effort":    _nan_none(r.control_effort),
            "steps":             r.steps,
            "replay_npz":        r.replay_npz,
        })

    def list_runs(self, benchmark_name: str | None = None) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(
                self.csv_path,
                usecols=["run_id", "timestamp", "benchmark", "dynamics", "quick"],
            ).drop_duplicates("run_id").sort_values("timestamp", ascending=False)
            if benchmark_name:
                df = df[df["benchmark"] == benchmark_name]
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def list_replay_results(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.csv_path)
            df = df[df["replay_npz"].notna() & (df["replay_npz"] != "")]
            return df[["run_id", "city", "algorithm", "segment",
                        "nominal_speed", "success", "replay_npz"]].copy()
        except Exception:
            return pd.DataFrame()

    def get_result(self, result_id: int) -> dict | None:
        return None

    def load_run(self, run_id: str) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.csv_path)
            return df[df["run_id"] == run_id].drop(
                columns=["run_id", "timestamp", "benchmark", "dynamics", "quick"],
                errors="ignore",
            ).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def export_csv(self, run_id: str, csv_path: Path) -> None:
        self.load_run(run_id).to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows_to_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        d = dict(r)
        records.append({
            "City":              d.get("city"),
            "Scenario":          d.get("scenario"),
            "Algorithm":         d.get("algorithm"),
            "Segment":           d.get("segment"),
            "Nominal Speed":     d.get("nominal_speed"),
            "Success":           bool(d.get("success")),
            "Is Clear":          bool(d.get("is_clear")),
            "Crash Reason":      d.get("crash_reason", ""),
            "Avg Solve Time":    d.get("avg_solve_time"),
            "Std Solve Time":    d.get("std_solve_time"),
            "Min Clearance":     d.get("min_clearance"),
            "Ref Min Clearance": d.get("ref_min_clearance"),
            "Travel Time":       d.get("travel_time"),
            "Path Length":       d.get("path_length"),
            "Converge Rate":     d.get("converge_rate"),
            "Control Effort":    d.get("control_effort"),
            "Steps":             d.get("steps"),
            "Replay NPZ":        d.get("replay_npz", ""),
        })
    return pd.DataFrame(records)