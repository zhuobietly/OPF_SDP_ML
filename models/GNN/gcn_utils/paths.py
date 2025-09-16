from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def data_dir() -> Path:
    return PROJECT_ROOT / "data"

def cache_dir() -> Path:
    p = PROJECT_ROOT / "data" / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p
