# registries.py
from typing import Dict, Callable, Any

# 六类注册表
_REGISTRIES: Dict[str, Dict[str, Any]] = {
    "reader": {},
    "dataset": {},
    "model": {},
    "loss": {},
    "evaluator": {},
    "task": {},
}

class RegistryError(KeyError):
    pass

def register(kind: str, name: str) -> Callable:
    """
    用法：
      @register("model", "GCN_arr_cls")
      class MyModel(nn.Module): ...
    """
    if kind not in _REGISTRIES:
        raise RegistryError(f"Unknown registry kind '{kind}'. "
                            f"Valid kinds: {list(_REGISTRIES.keys())}")

    def decorator(obj: Any) -> Any:
        if name in _REGISTRIES[kind]:
            raise RegistryError(f"Duplicate registration: {kind}:{name} 已存在")
        _REGISTRIES[kind][name] = obj
        return obj
    return decorator

def build(kind: str, name: str, **cfg) -> Any:
    """通过名字构建组件：build('model','GCN_arr_cls', hidden=64, ...)"""
    try:
        cls_or_fn = _REGISTRIES[kind][name]
    except KeyError as e:
        available = ", ".join(sorted(_REGISTRIES[kind].keys())) or "(空)"
        raise RegistryError(
            f"未找到 {kind}:{name}. 可用项：{available}"
        ) from e
    # 既支持类（需要实例化），也支持无参/有参工厂函数
    try:
        return cls_or_fn(**cfg)
    except TypeError:
        # 可能是无参数的可调用
        return cls_or_fn()

def available(kind: str):
    return sorted(_REGISTRIES[kind].keys())
