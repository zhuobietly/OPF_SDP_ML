# graphread.py
import re
import json
import numpy as np
from typing import Dict, Tuple, List, Optional

# ---------- 原有的辅助函数 ----------
def _strip_comments(line: str) -> str:
    i = line.find('%')
    return line if i < 0 else line[:i]

def _extract_numeric_block(text: str, key: str) -> np.ndarray:
    m = re.search(rf'mpc\.{re.escape(key)}\s*=\s*\[', text)
    if not m:
        return np.empty((0, 0), dtype=np.float64)
    start = m.end()
    m_end = re.search(r'\];', text[start:])
    if not m_end:
        raise ValueError(f"Block for mpc.{key} not closed with '];'")
    block = text[start:start + m_end.start()]

    rows = []
    for raw in block.splitlines():
        line = _strip_comments(raw).strip()
        if not line:
            continue
        if line.endswith(';'):
            line = line[:-1]
        parts = line.strip().split()
        if not parts:
            continue
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            continue
        rows.append(nums)

    if not rows:
        return np.empty((0, 0), dtype=np.float64)

    maxc = max(len(r) for r in rows)
    arr = np.zeros((len(rows), maxc), dtype=np.float64)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r
    return arr

def _extract_bus_names(text: str) -> List[str]:
    m = re.search(r'mpc\.bus_name\s*=\s*\{', text)
    if not m:
        return []
    start = m.end()
    m_end = re.search(r'\};', text[start:])
    if not m_end:
        raise ValueError("Block for mpc.bus_name not closed with '};'")
    block = text[start:start + m_end.start()]
    names = re.findall(r"'([^']*)'", block)
    return names

def _extract_baseMVA(text: str) -> float:
    m = re.search(r'mpc\.baseMVA\s*=\s*([0-9eE\+\-\.]+)\s*;', text)
    return float(m.group(1)) if m else 100.0

def load_matpower_case(path: str) -> Dict[str, object]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    text = "\n".join(_strip_comments(ln) for ln in raw.splitlines())

    bus = _extract_numeric_block(text, "bus")
    gen = _extract_numeric_block(text, "gen")
    branch = _extract_numeric_block(text, "branch")
    baseMVA = _extract_baseMVA(text)
    bus_name = _extract_bus_names(raw)
    return {"bus": bus, "gen": gen, "branch": branch,
            "baseMVA": baseMVA, "bus_name": bus_name}

def normalize_adj(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
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
    weight: str = "binary",
    use_status: bool = True,
    undirected: bool = True,
    eps: float = 1e-6
) -> np.ndarray:
    """
    MATPOWER branch 列（v2 常见）：
      0 fbus, 1 tbus, 2 r, 3 x, 4 b, ..., 8 ratio, 9 angle, 10 status, ...
    """
    A = np.zeros((n_buses, n_buses), dtype=np.float32)
    for row in branch:
        if row.size < 2:
            continue
        fbus, tbus = int(row[0]) - 1, int(row[1]) - 1
        if not (0 <= fbus < n_buses and 0 <= tbus < n_buses):
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
            A[fbus, tbus] = w
            if undirected:
                A[tbus, fbus] = w
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
    if include is None:
        include = dict(Pd=True, Qd=True, Gs=True, Bs=True, Vm=True, Va=True,
                       baseKV=True, type_onehot=True, has_gen=True, degree=True)

    N = bus.shape[0]
    feats = []
    colmap: Dict[str, slice] = {}

    def _add(name: str, arr: np.ndarray):
        nonlocal feats, colmap
        start = sum(f.shape[1] for f in feats)
        s = slice(start, start + arr.shape[1])
        feats.append(arr.astype(np.float32))
        colmap[name] = s

    if include.get("Pd", True):      _add("Pd", bus[:, 2:3])
    if include.get("Qd", True):      _add("Qd", bus[:, 3:4])
    if include.get("Gs", False):     _add("Gs", bus[:, 4:5])
    if include.get("Bs", False):     _add("Bs", bus[:, 5:6])
    if include.get("Vm", True):      _add("Vm", bus[:, 7:8])
    if include.get("Va", True):      _add("Va", bus[:, 8:9])
    if include.get("baseKV", True):  _add("baseKV", bus[:, 9:10])

    if include.get("type_onehot", True):
        oh = _one_hot_type(bus[:, 1])
        _add("type_onehot", oh)

    if include.get("has_gen", True):
        has_gen = np.zeros((N, 1), dtype=np.float32)
        if gen is not None and gen.size > 0:
            for row in gen:
                b = int(row[0]) - 1
                ok = True
                if row.size > 8:
                    try:
                        ok = int(row[7]) == 1  # 常见位置
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


# ---------- 面向对象封装 ----------
class GraphRead:
    """
    用法：
        gr = GraphRead("case14.m", weight="binary", add_self_loops=True)
        # 共用一份 A_hat
        X1 = gr.X_update(".../case14_0.05_perturbation_123_1.json")   # 刷新 Pd/Qd
        X2 = gr.X_update(".../case14_0.05_perturbation_123_2.json")

        # 需要改权重或是否加自环：
        gr.A_hat_gen(weight="inv_reactance", add_self_loops=False)
    """

    def __init__(
        self,
        case_path: str,
        weight: str = "binary",
        add_self_loops: bool = True,
        include_feats: Optional[Dict[str, bool]] = None
    ) -> None:
        # 解析 MATPOWER
        mpc = load_matpower_case(case_path)
        self.bus = mpc["bus"]
        self.gen = mpc["gen"]
        self.branch = mpc["branch"]
        self.baseMVA = mpc["baseMVA"]
        self.bus_name = mpc["bus_name"]
        self.N = int(self.bus.shape[0])

        # 初始邻接与规格化邻接
        self.weight = weight
        self.add_self_loops = add_self_loops
        self.A = build_adjacency_from_branch(self.branch, n_buses=self.N,
                                             weight=self.weight, use_status=True, undirected=True)
        self.A_hat = normalize_adj(self.A, add_self_loops=self.add_self_loops)

        # 初始特征（基于 .m 内自带的 Pd/Qd）
        self.include_feats = include_feats
        self.X, self.feature_index = build_node_features_from_bus_gen(
            self.bus, self.gen, self.A, include=self.include_feats
        )

        # info 字典对齐你的接口
        self.info: Dict[str, object] = dict(
            bus=self.bus, gen=self.gen, branch=self.branch,
            baseMVA=self.baseMVA, bus_name=self.bus_name,
            feature_index=self.feature_index
        )

        # 最近一次更新的场景编号（从文件名尾部提取）
        self.last_scenario_id: Optional[int] = None

    # 生成/更新 A_hat
    def A_hat_gen(self, weight: Optional[str] = None,
                  add_self_loops: Optional[bool] = None,
                  use_status: bool = True,
                  undirected: bool = True) -> np.ndarray:
        """
        重新生成 A 与 A_hat（若传入新参数则覆盖当前设置）。
        返回新的 A_hat，并更新 self.A, self.A_hat。
        """
        if weight is not None:
            self.weight = weight
        if add_self_loops is not None:
            self.add_self_loops = add_self_loops
        self.A = build_adjacency_from_branch(self.branch, n_buses=self.N,
                                             weight=self.weight,
                                             use_status=use_status,
                                             undirected=undirected)
        self.A_hat = normalize_adj(self.A, add_self_loops=self.add_self_loops)
        return self.A_hat

    # 刷新/生成 X（从 JSON 负荷文件替换 Pd/Qd）
    def X_update(self, load_json_path: str, inplace: bool = True) -> np.ndarray:
        """
        从 PowerModels 风格的负荷 JSON（{"load": {id: {"load_bus": b, "pd":..., "qd":...}}}）中
        汇总每个 bus 的 Pd/Qd，并替换当前 X 的 Pd/Qd 列。
        若 inplace=True，会更新 self.X 并返回它；否则返回一份副本。
        """
        with open(load_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        loads = obj.get("load", {})

        # 汇总到 bus（注意 bus 号是 1-based）
        Pd_bus = np.zeros((self.N,), dtype=np.float32)
        Qd_bus = np.zeros((self.N,), dtype=np.float32)

        for _, ld in loads.items():
            # 兼容两种来源：load_bus 或 source_id=["bus", bus]
            bus_idx = None
            if "load_bus" in ld:
                bus_idx = int(ld["load_bus"])
            elif "source_id" in ld and isinstance(ld["source_id"], list) and len(ld["source_id"]) >= 2:
                if str(ld["source_id"][0]).lower() == "bus":
                    bus_idx = int(ld["source_id"][1])
            if bus_idx is None:
                continue
            b0 = bus_idx - 1  # 转 0-based
            if 0 <= b0 < self.N:
                Pd_bus[b0] += float(ld.get("pd", 0.0))
                Qd_bus[b0] += float(ld.get("qd", 0.0))

        # 找到 Pd/Qd 在 X 里的列切片
        Pd_slice = self.feature_index.get("Pd", None)
        Qd_slice = self.feature_index.get("Qd", None)
        if Pd_slice is None or Qd_slice is None:
            raise RuntimeError("当前 X 中没有 Pd/Qd 列（检查 include_feats 或特征构造）。")

        X_new = self.X.copy()
        X_new[:, Pd_slice] = Pd_bus.reshape(-1, 1)
        X_new[:, Qd_slice] = Qd_bus.reshape(-1, 1)

        # 记录从文件名里解析到的场景编号（尾部 _<id>.json）
        self.last_scenario_id = _try_parse_suffix_id(load_json_path)
        # 也写到 info 里，便于外部取用
        self.info["scenario_id"] = self.last_scenario_id
        self.info["scenario_file"] = load_json_path

        if inplace:
            self.X = X_new
            return self.X
        else:
            return X_new

    # 和你需求对齐的名称别名（避免拼写带来的不便）
    def X_updata(self, load_json_path: str, inplace: bool = True) -> np.ndarray:
        return self.X_update(load_json_path, inplace=inplace)


# ---------- 小工具：从文件名尾部提取 _<id>.json 作为场景编号 ----------
def _try_parse_suffix_id(path: str) -> Optional[int]:
    """
    e.g. ".../case14_0.05_perturbation_123_17.json" -> 17
         ".../pure_p_02.json" -> None（没有尾部数字）
    """
    m = re.search(r'_(\d+)\.json$', path)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# ---------- 快速测试 ----------
if __name__ == "__main__":
    # 1) 初始化（只读一次 .m）
    gr = GraphRead("./data/raw_data/case14.m", weight="binary", add_self_loops=True)
    print("A_hat:", gr.A_hat.shape, "X:", gr.X.shape)

    # 2) 用不同的负荷 JSON 反复刷新 X（A_hat 复用）
    # json_path = "./data/scenarios/case14_0.05_perturbation_301_1.json"
    # X1 = gr.X_update(json_path)
    # print("scenario id:", gr.last_scenario_id, "X shape:", X1.shape)

    # 3) 重建 A_hat（可选）
    # gr.A_hat_gen(weight="inv_reactance", add_self_loops=False)
