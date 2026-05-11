"""
Unified CLI for the Trajectory Planner Benchmark system.

Usage:
  python cli.py run
  python cli.py run --quick --only Quadcopter
  python cli.py run --sanity
  python cli.py analyze
  python cli.py replay --list
  python cli.py replay <result_id>
  python cli.py list
  python cli.py export <run_id>
  python cli.py check
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    print("Error: Typer is not installed. Please install it using 'pip install typer'")
    sys.exit(1)

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root.parent))
sys.path.insert(0, str(_root))

app = typer.Typer(help="Trajectory Planner Benchmark Suite", no_args_is_help=True)


@app.command()
def run(
    config:   str           = typer.Option("benchmark.toml", "--config"),
    only:     Optional[str] = typer.Option(None,  "--only",     help="Run only the named benchmark"),
    quick:    bool          = typer.Option(False,  "--quick",    help="Quick mode (fewer cities/segments)"),
    sanity:   bool          = typer.Option(False,  "--sanity",   help="Run WMR sanity check"),
    no_store: bool          = typer.Option(False,  "--no-store", help="Disable result persistence"),
):
    """Run trajectory planner benchmarks."""
    from core import runner
    runner.run_cli(config, only, quick, no_store=no_store, sanity=sanity)


@app.command()
def analyze(
    config: str           = typer.Option("benchmark.toml", "--config"),
    only:   Optional[str] = typer.Option(None,     "--only",   help="Analyze only the named benchmark"),
    run_id: Optional[str] = typer.Option(None,     "--run-id", help="Analyze a single run by ID"),
    fmt:    str           = typer.Option("figure",  "--format", help="Output format: figure or json"),
):
    """Analyze benchmark results and generate figures or JSON summaries."""
    from analysis import analyze as analyzer
    analyzer.analyze_cli(config, only, run_id=run_id, fmt=fmt)


@app.command(name="replay")
def replay_cmd(
    result_id:  Optional[int]   = typer.Argument(None,  help="Result ID to replay (from 'replay --list')"),
    config:     str             = typer.Option("benchmark.toml", "--config"),
    speed:      Optional[float] = typer.Option(None,  "--speed", help="Playback speed multiplier"),
    save:       Optional[str]   = typer.Option(None,  "--save",  help="Save replay as video"),
    list_files: bool            = typer.Option(False, "--list",  help="List available replay results"),
):
    """Replay a saved benchmark flight by database result ID."""
    from core import replay as replayer
    replayer.replay_cli(result_id, config, speed, save, list_files)


@app.command(name="list")
def list_runs(
    config:    str           = typer.Option("benchmark.toml", "--config"),
    benchmark: Optional[str] = typer.Option(None, "--benchmark", help="Filter by benchmark name"),
):
    """List past benchmark runs stored in the database or CSV."""
    from data.config import load_config
    from data.store import open_store

    cfg        = load_config(Path(config))
    store_path = cfg.get("output", {}).get("store", "results/benchmark.db")

    if not Path(store_path).exists():
        print(f"No store found at '{store_path}'. Run a benchmark first.")
        raise typer.Exit(0)

    store = open_store(store_path)
    df    = store.list_runs(benchmark)

    if df.empty:
        print("No runs found.")
        raise typer.Exit(0)

    df["quick"] = df["quick"].map({0: "", 1: "quick"})
    df = df.rename(columns={
        "run_id": "Run ID", "timestamp": "Timestamp",
        "benchmark": "Benchmark", "dynamics": "Dynamics",
        "quick": "Mode", "notes": "Notes",
    })
    print(df.to_string(index=False))


@app.command()
def export(
    run_id: str            = typer.Argument(...,  help="Run ID to export (from 'list')"),
    output: Optional[str]  = typer.Option(None,  "--output", "-o", help="Output CSV path"),
    config: str            = typer.Option("benchmark.toml", "--config"),
):
    """Export a past benchmark run to CSV."""
    from data.config import load_config
    from data.store import open_store

    cfg        = load_config(Path(config))
    store_path = cfg.get("output", {}).get("store", "results/benchmark.db")

    if not Path(store_path).exists():
        print(f"No store found at '{store_path}'.")
        raise typer.Exit(1)

    store    = open_store(store_path)
    csv_path = Path(output) if output else Path(f"{run_id}.csv")
    store.export_csv(run_id, csv_path)
    print(f"Exported run '{run_id}' to {csv_path}")


@app.command()
def check(
    config: str = typer.Option("benchmark.toml", "--config"),
):
    """Validate the benchmark configuration and show a summary."""
    from data.config import load_config, validate_config

    cfg_path = Path(config)
    if not cfg_path.exists():
        print(f"Error: Configuration file '{config}' not found.")
        raise typer.Exit(1)

    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        print(f"Error parsing configuration: {e}")
        raise typer.Exit(1)

    errors = validate_config(cfg)
    if errors:
        print(f"Configuration '{config}' has errors:")
        for e in errors:
            print(f"  [ERR] {e}")
        raise typer.Exit(1)

    benchmarks   = cfg.get("benchmark", [])
    active       = [b for b in benchmarks if b.get("enabled", True)]
    run_cfg      = cfg.get("run", {})
    output_cfg   = cfg.get("output", {})
    n_cities     = len(cfg.get("city_registry", {}))
    n_vehicles   = len(cfg.get("vehicle_registry", {}))

    print(f"Configuration '{config}' OK")
    print(f"  Cities loaded    : {n_cities}")
    print(f"  Vehicles loaded  : {n_vehicles}")
    print(f"  Benchmarks       : {len(benchmarks)} total, {len(active)} enabled")
    print(f"  Max workers      : {run_cfg.get('max_workers', 6)}")
    print(f"  Clearance thresh : {run_cfg.get('clearance_threshold', 0.1)} m")
    print(f"  Save replays     : {run_cfg.get('save_replay', True)}")
    print(f"  Store            : {output_cfg.get('store', 'results/benchmark.db')}")

    if active:
        print("\nEnabled benchmarks:")
        for b in active:
            name    = b.get("name", "?")
            vehicle = b.get("vehicle", "?")
            algos   = b.get("algorithms", [])
            sc      = b.get("solver", {})
            cities  = b.get("cities", [])
            speeds  = b.get("nominal_speeds", [])
            print(f"  [{vehicle}] {name}")
            print(f"         algorithms  : {len(algos)}")
            print(f"         dt/horizon  : {sc.get('dt')}s / {sc.get('horizon_sec')}s")
            print(f"         cities      : {cities}")
            print(f"         speeds      : {speeds}")


if __name__ == "__main__":
    app()