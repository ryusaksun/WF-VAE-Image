#!/bin/bash

# ============================================
# WFVAE2Image 训练脚本（对齐 train_wfivae.sh）
# ============================================
#
# 使用方法：
# 1. 默认单卡训练：
#    bash train_wfimagevae.sh
#
# 2. 指定 GPU / 多卡：
#    GPU=3 bash train_wfimagevae.sh
#    GPU=0,1,2,3 bash train_wfimagevae.sh
#
# 3. 从 checkpoint 恢复：
#    RESUME_CKPT=/path/to/checkpoint.ckpt bash train_wfimagevae.sh
#
# 4. 指定原始 manifest 与划分比例：
#    ORIGINAL_MANIFEST=/path/to/images.jsonl TRAIN_RATIO=0.8 bash train_wfimagevae.sh
#
# 5. 覆盖训练步频参数：
#    EVAL_STEPS=500 SAVE_CKPT_STEP=1000 MAX_STEPS=50000 bash train_wfimagevae.sh
#
# ============================================

set -euo pipefail

# ============================================
# 可选：Conda 环境激活（按需取消注释）
# ============================================
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate wfvae

is_true() {
    local v
    v="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
    [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "y" ]]
}

resolve_master_port() {
    local requested_port="$1"
    local min_port="$2"
    local max_port="$3"

    python3 - "$requested_port" "$min_port" "$max_port" <<'PY'
import random
import socket
import sys

requested = sys.argv[1].strip()
min_port = int(sys.argv[2])
max_port = int(sys.argv[3])

if min_port < 1024 or max_port > 65535 or min_port > max_port:
    raise SystemExit("invalid-port-range")


def is_port_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


if requested:
    req = int(requested)
    if req < 1024 or req > 65535:
        raise SystemExit("invalid-requested-port")
    if is_port_free(req):
        print(req)
        raise SystemExit(0)

    # 请求端口被占用时，从 req+1 开始在区间内寻找空闲端口并回绕。
    search = list(range(req + 1, max_port + 1)) + list(range(min_port, req))
else:
    search = list(range(min_port, max_port + 1))
    random.shuffle(search)

for port in search:
    if is_port_free(port):
        print(port)
        raise SystemExit(0)

raise SystemExit("no-free-port")
PY
}

# ============================================
# GPU 配置
# ============================================
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
NUM_GPUS="$(echo "$GPU" | tr ',' '\n' | wc -l | tr -d ' ')"
echo "检测到 $NUM_GPUS 个 GPU: $GPU"

# ============================================
# NCCL / 线程配置
# ============================================
# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=bond0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# ============================================
# WandB 配置
# ============================================
export WANDB_PROJECT="${WANDB_PROJECT:-WFVAE-IMAGE}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"
if is_true "$DISABLE_WANDB"; then
    export WANDB_MODE=disabled
    WANDB_STATUS="关闭"
else
    unset WANDB_MODE || true
    WANDB_STATUS="开启 (project: ${WANDB_PROJECT})"
fi

# ============================================
# 路径与划分配置（对齐参考脚本）
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$SCRIPT_DIR")"

DEFAULT_MANIFEST="${SCRIPT_DIR}/ssk_image_manifest.jsonl"
ORIGINAL_MANIFEST="${ORIGINAL_MANIFEST:-$DEFAULT_MANIFEST}"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/sdc/${PROJECT_NAME}}"
TRAIN_MANIFEST="${OUTPUT_DIR}/train_manifest.jsonl"
EVAL_MANIFEST="${OUTPUT_DIR}/eval_manifest.jsonl"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"

# ============================================
# 训练配置
# ============================================
RESOLUTION="${RESOLUTION:-1024}"
MODEL_NAME="${MODEL_NAME:-WFVAE2Image}"
MODEL_CONFIG="${MODEL_CONFIG:-examples/wfvae2-image-1024.json}"
EXP_NAME="${EXP_NAME:-WFVAE2IMAGE-${RESOLUTION}}"

if [[ "$RESOLUTION" == "1024" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-4}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
else
    BATCH_SIZE="${BATCH_SIZE:-4}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
fi

GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EPOCHS="${EPOCHS:-100000}"
MAX_STEPS="${MAX_STEPS:-1000000}"
SAVE_CKPT_STEP="${SAVE_CKPT_STEP:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-64}"
EVAL_NUM_IMAGE_LOG="${EVAL_NUM_IMAGE_LOG:-8}"
LOG_STEPS="${LOG_STEPS:-10}"
CSV_LOG_STEPS="${CSV_LOG_STEPS:-200}"
CSV_LOG_PATH="${CSV_LOG_PATH:-}"
DISABLE_CSV_LOG="${DISABLE_CSV_LOG:-0}"
ENABLE_LIVE_PLOT="${ENABLE_LIVE_PLOT:-1}"
LIVE_PLOT_EVERY_STEPS="${LIVE_PLOT_EVERY_STEPS:-200}"
LIVE_PLOT_PATH="${LIVE_PLOT_PATH:-}"
LIVE_PLOT_HISTORY="${LIVE_PLOT_HISTORY:-0}"
LIVE_PLOT_DPI="${LIVE_PLOT_DPI:-150}"
ENABLE_PATCH_SCORE_VIS="${ENABLE_PATCH_SCORE_VIS:-1}"
PATCH_SCORE_VIS_MAX_SAMPLES="${PATCH_SCORE_VIS_MAX_SAMPLES:-0}"
PATCH_SCORE_MIN_CELL_PX="${PATCH_SCORE_MIN_CELL_PX:-32}"
PATCH_SCORE_SAVE_MAPS="${PATCH_SCORE_SAVE_MAPS:-1}"
PATCH_SCORE_SAVE_SUMMARY="${PATCH_SCORE_SAVE_SUMMARY:-1}"
PATCH_SCORE_LOG_WANDB="${PATCH_SCORE_LOG_WANDB:-1}"
ENABLE_VAL_IMAGE_DUMP="${ENABLE_VAL_IMAGE_DUMP:-1}"
VAL_IMAGE_DUMP_MAX_SAMPLES="${VAL_IMAGE_DUMP_MAX_SAMPLES:-0}"
DATASET_NUM_WORKER="${DATASET_NUM_WORKER:-8}"

DISC_CLS="${DISC_CLS:-wfimagevae.model.losses.LPIPSWithDiscriminator2D}"
DISC_START="${DISC_START:-5}"
DISC_WEIGHT="${DISC_WEIGHT:-0.5}"
KL_WEIGHT="${KL_WEIGHT:-1e-6}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-1.0}"
LOSS_TYPE="${LOSS_TYPE:-l1}"
LOGVAR_INIT="${LOGVAR_INIT:-0.0}"
WAVELET_WEIGHT="${WAVELET_WEIGHT:-0.1}"

MIX_PRECISION="${MIX_PRECISION:-bf16}"
CLIP_GRAD_NORM="${CLIP_GRAD_NORM:-1e5}"
EMA_DECAY="${EMA_DECAY:-0.999}"

EMA="${EMA:-1}"
FREEZE_ENCODER="${FREEZE_ENCODER:-0}"
FIND_UNUSED_PARAMETERS="${FIND_UNUSED_PARAMETERS:-0}"
IGNORE_MISMATCHED_SIZES="${IGNORE_MISMATCHED_SIZES:-0}"
WAVELET_LOSS="${WAVELET_LOSS:-1}"
NOT_RESUME_DISCRIMINATOR="${NOT_RESUME_DISCRIMINATOR:-0}"
NOT_RESUME_OPTIMIZER="${NOT_RESUME_OPTIMIZER:-0}"

RESUME_CKPT="${RESUME_CKPT:-}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT_MIN="${MASTER_PORT_MIN:-29500}"
MASTER_PORT_MAX="${MASTER_PORT_MAX:-29999}"
REQUESTED_MASTER_PORT="${MASTER_PORT:-}"
if [[ -n "$REQUESTED_MASTER_PORT" ]]; then
    MASTER_PORT="$(resolve_master_port "$REQUESTED_MASTER_PORT" "$MASTER_PORT_MIN" "$MASTER_PORT_MAX")"
    if [[ "$MASTER_PORT" != "$REQUESTED_MASTER_PORT" ]]; then
        echo "[WARN] 请求端口 ${REQUESTED_MASTER_PORT} 被占用，自动切换到 ${MASTER_PORT}"
    fi
else
    MASTER_PORT="$(resolve_master_port "" "$MASTER_PORT_MIN" "$MASTER_PORT_MAX")"
fi

# ============================================
# 函数：划分 manifest
# ============================================
split_dataset() {
    echo "================================================"
    echo "划分数据集 (训练集比例: ${TRAIN_RATIO})"
    echo "================================================"

    python3 <<PY
import random

manifest = r"${ORIGINAL_MANIFEST}"
train_manifest = r"${TRAIN_MANIFEST}"
eval_manifest = r"${EVAL_MANIFEST}"
train_ratio = float(r"${TRAIN_RATIO}")

with open(manifest, "r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]

total = len(lines)
if total == 0:
    raise RuntimeError(f"manifest 为空: {manifest}")

random.seed(42)
indices = list(range(total))
random.shuffle(indices)
split_idx = int(total * train_ratio)

train_lines = [lines[i] for i in indices[:split_idx]]
eval_lines = [lines[i] for i in indices[split_idx:]]

with open(train_manifest, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(eval_manifest, "w", encoding="utf-8") as f:
    f.writelines(eval_lines)

print(f"总样本数: {total}")
print(f"训练集: {len(train_lines)} -> {train_manifest}")
print(f"验证集: {len(eval_lines)} -> {eval_manifest}")
PY
}

cleanup() {
    if is_true "${KEEP_SPLIT_MANIFEST:-0}"; then
        echo "保留临时划分文件: ${TRAIN_MANIFEST}, ${EVAL_MANIFEST}"
        return
    fi

    echo ""
    echo "================================================"
    echo "清理临时划分文件"
    echo "================================================"

    [[ -f "$TRAIN_MANIFEST" ]] && rm -f "$TRAIN_MANIFEST" && echo "已删除: $TRAIN_MANIFEST"
    [[ -f "$EVAL_MANIFEST" ]] && rm -f "$EVAL_MANIFEST" && echo "已删除: $EVAL_MANIFEST"
}

trap cleanup EXIT

# ============================================
# 检查文件与目录
# ============================================
if [[ ! -f "$ORIGINAL_MANIFEST" ]]; then
    echo "[ERROR] 原始 manifest 不存在: $ORIGINAL_MANIFEST"
    echo "请设置 ORIGINAL_MANIFEST=/path/to/manifest.jsonl"
    exit 1
fi

if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "[ERROR] 模型配置文件不存在: $MODEL_CONFIG"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================
# 配置回显
# ============================================
echo "================================================"
echo "WFVAE2Image Training"
echo "================================================"
echo "原始 manifest: $ORIGINAL_MANIFEST"
echo "输出目录: $OUTPUT_DIR"
echo "模型配置: $MODEL_CONFIG"
echo "实验名: $EXP_NAME"
echo "分辨率: ${RESOLUTION}x${RESOLUTION}"
echo "训练集比例: $TRAIN_RATIO"
echo "GPU: $GPU ($NUM_GPUS 卡)"
echo "最大训练步数: $MAX_STEPS"
echo "Checkpoint 间隔: $SAVE_CKPT_STEP"
echo "验证间隔: $EVAL_STEPS"
echo "CSV 记录频率: ${CSV_LOG_STEPS} step(s)"
if is_true "$DISABLE_CSV_LOG"; then
    echo "CSV 记录: 关闭"
else
    if [[ -n "$CSV_LOG_PATH" ]]; then
        echo "CSV 路径: $CSV_LOG_PATH"
    else
        echo "CSV 路径: 自动(实验目录/training_losses.csv)"
    fi
fi
if is_true "$ENABLE_LIVE_PLOT"; then
    echo "实时损失图: 开启 (每 ${LIVE_PLOT_EVERY_STEPS} step 刷新)"
    if [[ -n "$LIVE_PLOT_PATH" ]]; then
        echo "实时损失图路径: $LIVE_PLOT_PATH"
    else
        echo "实时损失图路径: 自动(实验目录/training_losses_live.png)"
    fi
    if [[ "$LIVE_PLOT_HISTORY" == "0" ]]; then
        echo "实时损失图历史: 全量"
    else
        echo "实时损失图历史: 最近 ${LIVE_PLOT_HISTORY} 点"
    fi
    echo "实时损失图 DPI: ${LIVE_PLOT_DPI}"
else
    echo "实时损失图: 关闭"
fi
if is_true "$ENABLE_PATCH_SCORE_VIS"; then
    if [[ "$PATCH_SCORE_VIS_MAX_SAMPLES" == "0" ]]; then
        echo "PatchGAN 可视化样本数: 跟随 eval_num_image_log"
    else
        echo "PatchGAN 可视化样本数: $PATCH_SCORE_VIS_MAX_SAMPLES"
    fi
    echo "PatchGAN 网格最小 cell 像素: $PATCH_SCORE_MIN_CELL_PX"
    echo "PatchGAN 输出 summary.csv: $PATCH_SCORE_SAVE_SUMMARY"
    echo "PatchGAN 输出 patch_vis: $PATCH_SCORE_SAVE_MAPS"
    echo "PatchGAN 上传 wandb 统计: $PATCH_SCORE_LOG_WANDB"
else
    echo "PatchGAN 判别器打分可视化: 关闭"
fi
if is_true "$ENABLE_VAL_IMAGE_DUMP"; then
    if [[ "$VAL_IMAGE_DUMP_MAX_SAMPLES" == "0" ]]; then
        echo "验证原图/重建图导出样本数: 跟随 eval_num_image_log"
    else
        echo "验证原图/重建图导出样本数: $VAL_IMAGE_DUMP_MAX_SAMPLES"
    fi
else
    echo "验证原图/重建图导出: 关闭"
fi
echo "WandB: $WANDB_STATUS"
if [[ "$EVAL_SUBSET_SIZE" == "0" ]]; then
    echo "验证样本数: 全量"
else
    echo "验证样本数: $EVAL_SUBSET_SIZE"
fi
echo "MASTER_PORT: $MASTER_PORT"
echo "================================================"

if [[ -n "$RESUME_CKPT" ]]; then
    echo "RESUME MODE: ON"
    echo "Checkpoint: $RESUME_CKPT"
else
    echo "RESUME MODE: OFF"
fi
echo "================================================"

# ============================================
# 划分数据集
# ============================================
split_dataset

# ============================================
# 启动训练
# ============================================
LOG_FILE="${OUTPUT_DIR}/training_wfimagevae_${RESOLUTION}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting training..."

CMD=(
    torchrun
    --nnodes=1
    --nproc_per_node="$NUM_GPUS"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    train_image_ddp.py
    --exp_name "$EXP_NAME"
    --image_path "$TRAIN_MANIFEST"
    --eval_image_path "$EVAL_MANIFEST"
    --use_manifest
    --model_name "$MODEL_NAME"
    --model_config "$MODEL_CONFIG"
    --ckpt_dir "$OUTPUT_DIR"
    --resolution "$RESOLUTION"
    --eval_resolution "$RESOLUTION"
    --eval_crop_size "$RESOLUTION"
    --batch_size "$BATCH_SIZE"
    --eval_batch_size "$EVAL_BATCH_SIZE"
    --grad_accum_steps "$GRAD_ACCUM_STEPS"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --epochs "$EPOCHS"
    --max_steps "$MAX_STEPS"
    --save_ckpt_step "$SAVE_CKPT_STEP"
    --eval_steps "$EVAL_STEPS"
    --eval_subset_size "$EVAL_SUBSET_SIZE"
    --eval_num_image_log "$EVAL_NUM_IMAGE_LOG"
    --mix_precision "$MIX_PRECISION"
    --disc_cls "$DISC_CLS"
    --disc_start "$DISC_START"
    --disc_weight "$DISC_WEIGHT"
    --kl_weight "$KL_WEIGHT"
    --perceptual_weight "$PERCEPTUAL_WEIGHT"
    --loss_type "$LOSS_TYPE"
    --logvar_init "$LOGVAR_INIT"
    --wavelet_weight "$WAVELET_WEIGHT"
    --clip_grad_norm "$CLIP_GRAD_NORM"
    --ema_decay "$EMA_DECAY"
    --log_steps "$LOG_STEPS"
    --csv_log_steps "$CSV_LOG_STEPS"
    --live_plot_every_steps "$LIVE_PLOT_EVERY_STEPS"
    --live_plot_history "$LIVE_PLOT_HISTORY"
    --live_plot_dpi "$LIVE_PLOT_DPI"
    --patch_score_vis_max_samples "$PATCH_SCORE_VIS_MAX_SAMPLES"
    --patch_score_min_cell_px "$PATCH_SCORE_MIN_CELL_PX"
    --val_image_dump_max_samples "$VAL_IMAGE_DUMP_MAX_SAMPLES"
    --dataset_num_worker "$DATASET_NUM_WORKER"
)

if [[ -n "$RESUME_CKPT" ]]; then
    CMD+=(--resume_from_checkpoint "$RESUME_CKPT")
fi

if [[ -n "$PRETRAINED_MODEL" ]]; then
    CMD+=(--pretrained_model_name_or_path "$PRETRAINED_MODEL")
fi

if is_true "$EMA"; then
    CMD+=(--ema)
fi

if is_true "$FREEZE_ENCODER"; then
    CMD+=(--freeze_encoder)
fi

if is_true "$FIND_UNUSED_PARAMETERS"; then
    CMD+=(--find_unused_parameters)
fi

if is_true "$IGNORE_MISMATCHED_SIZES"; then
    CMD+=(--ignore_mismatched_sizes)
fi

if is_true "$WAVELET_LOSS"; then
    CMD+=(--wavelet_loss)
fi

if is_true "$NOT_RESUME_DISCRIMINATOR"; then
    CMD+=(--not_resume_discriminator)
fi

if is_true "$NOT_RESUME_OPTIMIZER"; then
    CMD+=(--not_resume_optimizer)
fi

if [[ -n "$CSV_LOG_PATH" ]]; then
    CMD+=(--csv_log_path "$CSV_LOG_PATH")
fi

if is_true "$DISABLE_CSV_LOG"; then
    CMD+=(--disable_csv_log)
fi

if [[ -n "$LIVE_PLOT_PATH" ]]; then
    CMD+=(--live_plot_path "$LIVE_PLOT_PATH")
fi

if ! is_true "$ENABLE_LIVE_PLOT"; then
    CMD+=(--disable_live_plot)
fi

if ! is_true "$ENABLE_PATCH_SCORE_VIS"; then
    CMD+=(--disable_patch_score_vis)
fi

if ! is_true "$PATCH_SCORE_SAVE_MAPS"; then
    CMD+=(--no_patch_score_save_maps)
fi

if ! is_true "$PATCH_SCORE_SAVE_SUMMARY"; then
    CMD+=(--no_patch_score_save_summary)
fi

if ! is_true "$PATCH_SCORE_LOG_WANDB"; then
    CMD+=(--no_patch_score_log_wandb)
fi

if ! is_true "$ENABLE_VAL_IMAGE_DUMP"; then
    CMD+=(--disable_val_image_dump)
fi

echo "执行命令:"
printf ' %q' "${CMD[@]}"
echo

echo "使用 torchrun 启动训练 (${NUM_GPUS} GPUs, port: ${MASTER_PORT})"
"${CMD[@]}"

echo ""
echo "================================================"
echo "Training completed!"
echo "输出目录: $OUTPUT_DIR"
echo "================================================"
