import hashlib, pickle
from .paths import cache_dir

def cfg_hash(obj) -> str:
    s = repr(sorted(obj.items())) if isinstance(obj, dict) else repr(obj)
    return hashlib.md5(s.encode()).hexdigest()[:10]

def save_pickle(obj, name:str):
    path = cache_dir() / f"{name}.pkl"
    with open(path, "wb") as f: pickle.dump(obj, f)
    return path

def load_pickle(name:str):
    path = cache_dir() / f"{name}.pkl"
    with open(path, "rb") as f: return pickle.load(f)
