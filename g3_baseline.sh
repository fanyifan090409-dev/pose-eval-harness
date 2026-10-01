#!/bin/bash
# G3 基准实验全量重跑（论文 A 锁定基准数）：双卡并行推断 → 离线引擎 → 门禁
# 用法: nohup bash g3_baseline.sh > /home/ubuntu/isef/logs/g3_master.log 2>&1 &
# 仅重跑收尾: bash g3_baseline.sh finalize
# 注意: 离线引擎 --run 传裸 run_id（引擎内部已拼 ROOT/runs/ 前缀）
set -u
PY=/home/ubuntu/Miniconda3/envs/py310/bin/python
cd /home/ubuntu/isef/eval-harness || exit 1
LOGDIR=/home/ubuntu/isef/logs

if [ "${1:-}" != "finalize" ]; then
  worker() {
    local model="$1" gpu="$2" ds batch
    local log="$LOGDIR/g3_${model}_gpu${gpu}.log"
    {
      echo "===== worker start $(date '+%F %T') model=$model gpu=$gpu ====="
      for ds in coco-val crowdpose-val ochuman-test posetrack-val; do
        batch=32
        [ "$ds" = "posetrack-val" ] && batch=64
        echo "----- [$ds] run_infer batch=$batch -----"
        if ! "$PY" harness/run_infer.py --model "$model" --dataset "$ds" --device "$gpu" --batch "$batch"; then
          echo "!! FAIL infer $model $ds"
        fi
      done
      echo "===== worker done $(date '+%F %T') model=$model ====="
    } >> "$log" 2>&1
  }

  worker round5-3 0 &
  P0=$!
  worker yolo11s-pose 1 &
  P1=$!
  wait "$P0" "$P1"
fi

FAIL=0
{
  echo "===== finalize start $(date '+%F %T') ====="
  for model in round5-3 yolo11s-pose; do
    for ds in coco-val crowdpose-val ochuman-test posetrack-val; do
      run="${model}__${ds}"
      [ -f "runs/$run/preds.json" ] || { echo "!! 缺 runs/$run/preds.json，跳过"; FAIL=1; continue; }
      echo "----- $run M1/M2/M3 -----"
      "$PY" harness/stratified_map.py --run "$run" || { echo "!! M1 fail $run"; FAIL=1; }
      "$PY" harness/angles.py --run "$run" || { echo "!! M2 fail $run"; FAIL=1; }
      if [ "$ds" = "posetrack-val" ]; then
        "$PY" harness/temporal.py --run "$run" || { echo "!! M3 fail $run"; FAIL=1; }
      fi
    done
  done
  for ds in coco-val crowdpose-val ochuman-test posetrack-val; do
    echo "----- stats $ds -----"
    "$PY" harness/statistics.py --dataset "$ds" \
      --run-a "yolo11s-pose__${ds}" --run-b "round5-3__${ds}" || { echo "!! stats fail $ds"; FAIL=1; }
  done
  if [ "$FAIL" -eq 0 ]; then
    echo "----- report -----"
    "$PY" harness/report.py || { echo "!! report fail"; FAIL=1; }
  else
    echo "!! 存在失败项，跳过 report 防止陈旧数据混入"
  fi
  echo "===== finalize done $(date '+%F %T') FAIL=$FAIL ====="
} > "$LOGDIR/g3_finalize.log" 2>&1

if [ "$FAIL" -eq 0 ]; then
  {
    echo "===== gates start $(date '+%F %T') ====="
    "$PY" tests/test_gates.py
    echo "===== gates done $(date '+%F %T') ====="
  } > "$LOGDIR/g3_gates.log" 2>&1
else
  echo "!! finalize 存在失败项，跳过门禁" > "$LOGDIR/g3_gates.log"
fi

echo "ALL DONE $(date '+%F %T') FAIL=$FAIL"
