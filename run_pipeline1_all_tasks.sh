#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录执行。
# 默认会按 task01 -> task09 顺序执行；也可通过 --tasks 覆盖。

DESIGNER_AGENT_CONFIG="agent_configs/designer_agent.json"
CRITIC_AGENT_CONFIG="agent_configs/critic_agent.json"
STEP_GENERATOR_AGENT_CONFIG="agent_configs/step_generator_agent.json"
EXEC_PYTHON=""
PYTHON_BIN="python"

TASKS=(
  task01_calabash
  task02_reaching
  task03_flatten
  task04_holder
  task05_waterfill
  task06_piggy
  task07_lifting
  task08_cutting
  task09_transport
)

usage() {
  cat <<USAGE
用法:
  bash run_pipeline1_all_tasks.sh [选项]

选项:
  --tasks "task01_calabash task02_reaching"
      指定要执行的任务列表（空格分隔）。
  --designer_agent_config <path>
  --critic_agent_config <path>
  --step_generator_agent_config <path>
  --exec_python <path>
      传给 pipeline1.py 的 --exec_python。
  --python <bin>
      用哪个 Python 解释器来启动 utils/pipeline1.py（默认: python）。
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      shift
      read -r -a TASKS <<< "${1:-}"
      ;;
    --designer_agent_config)
      shift
      DESIGNER_AGENT_CONFIG="${1:-}"
      ;;
    --critic_agent_config)
      shift
      CRITIC_AGENT_CONFIG="${1:-}"
      ;;
    --step_generator_agent_config)
      shift
      STEP_GENERATOR_AGENT_CONFIG="${1:-}"
      ;;
    --exec_python)
      shift
      EXEC_PYTHON="${1:-}"
      ;;
    --python)
      shift
      PYTHON_BIN="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

for task_name in "${TASKS[@]}"; do
  task_prompt_json_dir="${PWD}/${task_name}/task_prompt.json"
  if [[ ! -f "$task_prompt_json_dir" ]]; then
    echo "[跳过] ${task_name}: 未找到 ${task_prompt_json_dir}"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" utils/pipeline1.py
    --task_name "$task_name"
    --task_prompt_json_dir "$task_prompt_json_dir"
    --designer_agent_config "$DESIGNER_AGENT_CONFIG"
    --critic_agent_config "$CRITIC_AGENT_CONFIG"
    --step_generator_agent_config "$STEP_GENERATOR_AGENT_CONFIG"
  )

  if [[ -n "$EXEC_PYTHON" ]]; then
    cmd+=(--exec_python "$EXEC_PYTHON")
  fi

  echo "========================================"
  echo "[运行] ${task_name}"
  printf '命令: '
  printf '%q ' "${cmd[@]}"
  echo
  "${cmd[@]}"
  echo "[完成] ${task_name}"
done

echo "全部任务执行结束。"
