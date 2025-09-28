#!/usr/bin/env bash
set -euo pipefail

# === 项目根：默认取脚本所在目录的上一层；也可以用 PROJECT_DIR=... 覆盖 ===
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"

# === 输入目录与算例名（可用环境变量覆盖） ===
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/data/load_profiles/case2746wop}"
CASE_NAME="${CASE_NAME:-case2746wop}"

# 要跑的配置
FMS=("Chordal_MD" "Chordal_MFI" "Chordal_AMD")
MERGINGS=("false" "true")
ALPHAS=("2.0" "3.0" "4.0" "5.0")

echo "[INFO] PROJECT_DIR = $PROJECT_DIR"
echo "[INFO] INPUT_DIR   = $INPUT_DIR"
echo "[INFO] CASE_NAME   = $CASE_NAME"

# 基本检查
if [ ! -d "$INPUT_DIR" ]; then
  echo "[FATAL] INPUT_DIR 不存在：$INPUT_DIR"
  echo "        请检查路径，或在执行时传入：INPUT_DIR=/abs/path/to/case2746wop"
  exit 2
fi

# 统计 json 数量
shopt -s nullglob
jsons=( "$INPUT_DIR"/*.json )
num_json=${#jsons[@]}
if [ "$num_json" -eq 0 ]; then
  echo "[FATAL] 在 $INPUT_DIR 里找不到 *.json"
  echo "        用 ls 看看：ls -l \"$INPUT_DIR\""
  exit 3
fi
echo "[INFO] 发现 JSON 文件数量：$num_json"

# 生成 tasks 文件
OUT_DIR="$PROJECT_DIR/task"
mkdir -p "$OUT_DIR"
OUT_FILE="$OUT_DIR/2746_0.07_tasks.tsv"
: > "$OUT_FILE"

for json in "${jsons[@]}"; do
  for fm in "${FMS[@]}"; do
    for m in "${MERGINGS[@]}"; do
      if [[ "$m" == "true" ]]; then
        for a in "${ALPHAS[@]}"; do
          echo "$CASE_NAME|$json|$fm|$m|$a" >> "$OUT_FILE"
        done
      else
        echo "$CASE_NAME|$json|$fm|$m|0.0" >> "$OUT_FILE"
      fi
    done
  done
done

echo "✅ 任务清单已生成：$(wc -l < "$OUT_FILE") 条（$OUT_FILE）"
