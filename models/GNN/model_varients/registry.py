REGISTRY = {}

def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco

def build(name: str, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
