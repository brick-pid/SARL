#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export WANDB_MODE=offline

# Data and Model
MODEL_PATH=/inspire/ssd/project/project-public/xinyi-253308120310/thinktool/slime/model/qwen3-4b-inst-2507
DATA_DIR=/inspire/ssd/project/project-public/xinyi-253308120310/thinktool/slime/data/agentgym
# ENV="alfworld"
# ENV="alfworld sciworld"
# ENV="alfworld sciworld webshop"
ENV=${ENV:-"alfworld"}

TRAIN_PATH=""
EVAL_PROMPT_DATA=()
for env in $ENV; do
   TRAIN_PATH="${TRAIN_PATH} ${DATA_DIR}/train/${env}_train.jsonl"
   EVAL_PROMPT_DATA+=("${env}" "${DATA_DIR}/test/${env}_test.jsonl")
done
TRAIN_PATH="${TRAIN_PATH# }"  # trim leading space

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"



SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
LOAD_SAVE_PATH="/root/shared_data/${RUN_ID}/checkpoints"

CKPT_ARGS=(
   --hf-checkpoint $MODEL_PATH
   --load $MODEL_PATH
   --ref-load $MODEL_PATH
)


ROLLOUT_ARGS=(
   --prompt-data $TRAIN_PATH
   --input-key task_id
   --metadata-key metadata
   --rollout-shuffle
   --balance-data
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-max-context-len 32768
   --rollout-temperature 1
   --global-batch-size 64
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data "${EVAL_PROMPT_DATA[@]}"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8192
   --eval-top-p 1
)

GRPO_ARGS=(
   --use-kl-loss
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

if [ -z "${WANDB_API_KEY}" ]; then
   WANDB_ARGS=()
else
   WANDB_ARGS=(
      --use-wandb
      --wandb-project slime-dev-mcore-fsdp
      --wandb-group qwen3-4B-fsdp-1130-ref
      --wandb-key "${WANDB_API_KEY}"
   )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.75
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-attention-backend fa3
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_3
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 2
   --colocate
   --use-fault-tolerance
   --dump-details /inspire/ssd/project/project-public/xinyi-253308120310/thinktool/slime/dump
   # --fsdp-cpu-offload
)

CUSTOM_ARGS=(
   --custom-generate-function-path experiments.generate.generate
   --custom-config-path /inspire/ssd/project/project-public/xinyi-253308120310/thinktool/slime/experiments/config.yaml
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}/..\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${TRAIN_BACKEND_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"

