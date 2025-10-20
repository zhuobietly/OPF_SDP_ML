from .registry import build, register, REGISTRY
from .gcn import GCN  # ensure registration
from .gcn import GCNGlobal  # ensure registration

__all__ = ["build", "register", "REGISTRY", "GCN","GCNGlobal"]
