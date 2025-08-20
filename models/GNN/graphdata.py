# matpower_to_gnn.py
import re
import numpy as np
from typing import Dict, Tuple, List

# ---------- 1) 基础工具：去注释、找数组/cell 块 ----------
def _strip_comments(line: str) -> str:
    # 去除 % 后的注释
    i = line.find('%')
    return line if i < 0 else line[:i]

def _extract_numeric_block(text: str, key: str) -> np.ndarray:
    """
    从文本中提取类似：
      mpc.<key> = [
         1  2  0.1  0.2 ...
         ...
      ];
    的数值块，返回 np.ndarray (rows, cols)
    """
    # 找开头
    m = re.search(rf'mpc\.{re.escape(key)}\s*=\s*\[', text)
    if not m:
        return np.empty((0, 0), dtype=np.float64)
    start = m.end()
    # 找结尾 '];'
    m_end = re.search(r'\];', text[start:])
    if not m_end:
        raise ValueError(f"Block for mpc.{key} not closed with '];'")
    block = text[start:start + m_end.start()]

    rows = []
    for raw in block.splitlines():
        line = _strip_comments(raw).strip()
        if not line:
            continue
        # 去掉行末分号
        if line.endswith(';'):
            line = line[:-1]
        parts = line.strip().split()
        if not parts:
            continue
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            # 有些行可能是空或包含无关字符，跳过
            continue
        rows.append(nums)

    if not rows:
        return np.empty((0, 0), dtype=np.float64)

    # 对齐列数
    maxc = max(len(r) for r in rows)
    arr = np.zeros((len(rows), maxc), dtype=np.float64)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r
    return arr

def _extract_bus_names(text: str) -> List[str]:
    """
    解析：
      mpc.bus_name = {
        'Bus 1     HV';
        'Bus 2     HV';
        ...
      };
    """
    m = re.search(r'mpc\.bus_name\s*=\s*\{', text)
    if not m:
        return []
    start = m.end()
    m_end = re.search(r'\};', text[start:])
    if not m_end:
        raise ValueError("Block for mpc.bus_name not closed with '};'")
    block = text[start:start + m_end.start()]
    # 匹配单引号里的字符串
    names = re.findall(r"'([^']*)'", block)
    return names

def _extract_baseMVA(text: str) -> float:
    m = re.search(r'mpc\.baseMVA\s*=\s*([0-9eE\+\-\.]+)\s*;', text)
    return float(m.group(1)) if m else 100.0

# ---------- 2) 读入 MATPOWER case 文件 ----------
def load_matpower_case(path: str) -> Dict[str, object]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    text = "\n".join(_strip_comments(ln) for ln in raw.splitlines())

    bus = _extract_numeric_block(text, "bus")
    gen = _extract_numeric_block(text, "gen")
    branch = _extract_numeric_block(text, "branch")
    baseMVA = _extract_baseMVA(text)
    bus_name = _extract_bus_names(raw)  # 名称保留原始文本解析（可能含空格）

    return {
        "bus": bus,               # (Nb, ?)
        "gen": gen,               # (Ng, ?)
        "branch": branch,         # (Nl, ?)
        "baseMVA": baseMVA,
        "bus_name": bus_name,     # list[str]
    }

# ---------- 3) 图构建与特征 ----------
def normalize_adj(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    "领接矩阵归一化：D^(-1/2) * A * D^(-1/2)"
    A = A.astype(np.float32, copy=False)
    if add_self_loops:
        A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(axis=1, keepdims=False)
    d_inv_sqrt = np.power(d + 1e-8, -0.5, dtype=np.float32)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

def build_adjacency_from_branch(
    branch: np.ndarray,
    n_buses: int,
    weight: str = "binary",      # ["binary", "inv_impedance", "inv_reactance"]
    use_status: bool = True,
    undirected: bool = True,
    eps: float = 1e-6
) -> np.ndarray:
    """
    MATPOWER branch 列（版本2常见）：
      0 fbus, 1 tbus, 2 r, 3 x, 4 b, ..., 8 ratio, 9 angle, 10 status, ...
    """
    A = np.zeros((n_buses, n_buses), dtype=np.float32)
    for row in branch:
        if row.size < 2:
            continue
        fbus, tbus = int(row[0]) - 1, int(row[1]) - 1  # 转 0-based
        if fbus < 0 or fbus >= n_buses or tbus < 0 or tbus >= n_buses:
            continue
        r = float(row[2]) if row.size > 2 else 0.0
        x = float(row[3]) if row.size > 3 else 0.0
        status = int(row[10]) if (use_status and row.size > 10) else 1
        if use_status and status == 0:
            continue

        if weight == "binary":
            w = 1.0
        elif weight == "inv_impedance":
            z = (r*r + x*x) ** 0.5
            w = 1.0 / (z + eps)
        elif weight == "inv_reactance":
            w = 1.0 / (abs(x) + eps)
        else:
            raise ValueError(f"Unknown weight='{weight}'")
        if A[fbus, tbus] == 0.0:
            A[fbus, tbus] += w
            if undirected:
                A[tbus, fbus] += w
        else:
            continue  # 避免重复边
    np.fill_diagonal(A, 0.0)
    return A

def _one_hot_type(bus_type: np.ndarray) -> np.ndarray:
    # 1=PQ, 2=PV, 3=Slack
    types = bus_type.astype(int)
    oh = np.zeros((len(types), 3), dtype=np.float32)
    idx = np.clip(types - 1, 0, 2)
    oh[np.arange(len(types)), idx] = 1.0
    return oh

def build_node_features_from_bus_gen(
    bus: np.ndarray,
    gen: np.ndarray,
    A: np.ndarray,
    include: Dict[str, bool] = None,
) -> Tuple[np.ndarray, Dict[str, slice]]:
    """
    生成 X，并返回每个特征在 X 中的列位置（便于之后替换 Pd/Qd）。
    bus 列：0 bus_i, 1 type, 2 Pd, 3 Qd, 4 Gs, 5 Bs, 6 area, 7 Vm, 8 Va, 9 baseKV, 10 zone, 11 Vmax, 12 Vmin
    gen 列：0 bus, 7 status（若存在）
    include: 选择要包含的字段（缺省全开）
    """
    if include is None:
        include = dict(Pd=True, Qd=True, Gs=True, Bs=True, Vm=True, Va=True,
                       baseKV=True, type_onehot=True, has_gen=True, degree=True)

    N = bus.shape[0]
    feats = []
    colmap: Dict[str, slice] = {}

    def _add(name: str, arr: np.ndarray):
        nonlocal feats, colmap
        s = slice(sum(f.shape[1] for f in feats), sum(f.shape[1] for f in feats) + arr.shape[1])
        feats.append(arr.astype(np.float32))
        colmap[name] = s

    if include.get("Pd", True):
        _add("Pd", bus[:, 2:3])
    if include.get("Qd", True):
        _add("Qd", bus[:, 3:4])
    if include.get("Gs", False):
        _add("Gs", bus[:, 4:5])
    if include.get("Bs", False):
        _add("Bs", bus[:, 5:6])
    if include.get("Vm", True):
        _add("Vm", bus[:, 7:8])
    if include.get("Va", True):
        _add("Va", bus[:, 8:9])
    if include.get("baseKV", True):
        _add("baseKV", bus[:, 9:10])

    if include.get("type_onehot", True):
        oh = _one_hot_type(bus[:, 1])
        _add("type_onehot", oh)

    if include.get("has_gen", True):
        has_gen = np.zeros((N, 1), dtype=np.float32)
        if gen is not None and gen.size > 0:
            for row in gen:
                b = int(row[0]) - 1
                ok = True
                if row.size > 8:  # 常见 gen 的 status 在第 8 列（index 7），但不同 case 可能列数不同
                    # 更稳妥：若存在第7列（index 7）作为 status
                    try:
                        ok = int(row[7]) == 1
                    except Exception:
                        ok = True
                if 0 <= b < N and ok:
                    has_gen[b, 0] = 1.0
        _add("has_gen", has_gen)

    if include.get("degree", True):
        deg = A.sum(axis=1, keepdims=True).astype(np.float32)
        _add("degree", deg)

    X = np.concatenate(feats, axis=1) if feats else np.zeros((N, 0), dtype=np.float32)
    return X, colmap

def prepare_gnn_io_from_matpower_file(
    path: str,
    weight: str = "binary",
    add_self_loops: bool = True,
    include_feats: Dict[str, bool] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    读 .m 文件 -> (A_hat, X, info)
      - A_hat: 规范化邻接 (N,N)
      - X:     节点特征 (N,F)
      - info:  { 'bus','gen','branch','baseMVA','bus_name','feature_index' }
    """
    mpc = load_matpower_case(path)
    bus, gen, branch = mpc["bus"], mpc["gen"], mpc["branch"]
    N = bus.shape[0]

    A = build_adjacency_from_branch(branch, n_buses=N, weight=weight, use_status=True, undirected=True)
    A_hat = normalize_adj(A, add_self_loops=add_self_loops)

    X, feat_idx = build_node_features_from_bus_gen(bus, gen, A, include=include_feats)

    info = dict(
        bus=bus, gen=gen, branch=branch,
        baseMVA=mpc["baseMVA"], bus_name=mpc["bus_name"],
        feature_index=feat_idx  # e.g. feat_idx['Pd'] gives slice in X
    )
    return A_hat, X, info

# ---------- 4) 快速测试 ----------
if __name__ == "__main__":
    # 举例：path = "case4.m" 或 "case14.m"
    path = "./data/raw_data/case14.m"
    A_hat, X, info = prepare_gnn_io_from_matpower_file(path, weight="binary", add_self_loops=True)
    print("A_hat:", A_hat.shape, "X:", X.shape)
    print("Feature columns:", {k: (v.start, v.stop) for k, v in info["feature_index"].items()})
