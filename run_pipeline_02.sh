#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 自动探测 pipeline v2: dataset/YOLO → SAM3 → Judge → Report → 失败图片再优化
#
# v2 新增: Step 4 — 对 fail/review 图片用 Qwen 分析 YOLO box 内容,
#          SAM3 混合提示 (box+text) 重分割, 重新 judge, 合并真值
#
# v2.1: 支持 model/ + production_data/ 多模型布局,
#       自动按名称匹配, 每个子物料独立 Instance_<name>/ 输出
#
# 用法:
#   bash run_pipeline_02.sh <input_dir> [output_dir]
#
# 可选环境变量:
#   SAM3_CHECKPOINT      SAM3 模型权重 (默认 /home/model/sam3_pth/sam3pt/sam3.pt)
#   JUDGE_MODEL_PATH      Qwen-VL 模型路径 (默认 /home/model/llm_model/qwen_4b)
#   MAX_IMAGES            judge 阶段最大图片数 (默认 0 = 全部)
#   JUDGE_PASS_THRESHOLD  合格分数阈值 (默认 0.75)
#   SAM3_CONDA_ENV        SAM3 conda 环境名 (默认 sam3_6000d)
#   JUDGE_CONDA_ENV       judge conda 环境名 (默认 vllm_new)
#   GPU_DEVICE            指定 GPU 卡号 (如 0 或 1), 不设则使用全部 GPU
#   SKIP_RERUN            设为 1 跳过 Step 4 再优化 (默认 0 = 执行)
# ============================================================

INPUT_DIR="${1:?用法: bash run_pipeline_02.sh <input_dir> [output_dir]}"

INPUT_DIR="${INPUT_DIR%/}"
FOLDER_NAME="$(basename "$INPUT_DIR")"
OUTPUT_DIR="${2:-${INPUT_DIR}/output_${FOLDER_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR%/}"

# ---- 可选环境变量默认值 ----
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/home/model/sam3_pth/sam3pt/sam3.pt}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/home/model/llm_model/qwen_4b}"
MAX_IMAGES="${MAX_IMAGES:-0}"
JUDGE_PASS_THRESHOLD="${JUDGE_PASS_THRESHOLD:-0.75}"
SAM3_CONDA_ENV="${SAM3_CONDA_ENV:-sam3_6000d}"
JUDGE_CONDA_ENV="${JUDGE_CONDA_ENV:-vllm_new}"
GPU_DEVICE="${GPU_DEVICE:-}"
SKIP_RERUN="${SKIP_RERUN:-0}"

# ---- GPU 绑定 ----
if [ -n "$GPU_DEVICE" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
fi

# ---- Step 0: 探测模式 ----
MULTI_MODEL=0
if [ -d "${INPUT_DIR}/dataset" ]; then
    MODE="dataset"
    SPLITS="train valid test"
    JUDGE_IMAGE_DIR="${INPUT_DIR}/dataset"
else
    MODE="yolo"
    PT_FILES=()
    while IFS= read -r -d '' f; do
        PT_FILES+=("$f")
    done < <(find "${INPUT_DIR}" -maxdepth 3 -name "*.pt" -type f -print0 2>/dev/null || true)

    if [ "${#PT_FILES[@]}" -eq 0 ]; then
        echo "[ERROR] 未找到 .pt 模型文件"
        exit 1
    fi

    BEST_PT_COUNT=0
    BEST_PT=""
    for f in "${PT_FILES[@]}"; do
        if [ "$(basename "$f")" = "best.pt" ]; then
            BEST_PT_COUNT=$((BEST_PT_COUNT + 1))
            [ "$BEST_PT_COUNT" -eq 1 ] && BEST_PT="$f"
        fi
    done

    if [ "$BEST_PT_COUNT" -eq 1 ]; then
        PT_FOUND="$BEST_PT"
    elif [ "$BEST_PT_COUNT" -gt 1 ]; then
        echo "[INFO] 发现 ${BEST_PT_COUNT} 个 best.pt，交由 batch_01.py model/production_data 自动匹配"
        for f in "${PT_FILES[@]}"; do echo "  $f"; done
        MULTI_MODEL=1
    elif [ "${#PT_FILES[@]}" -eq 1 ]; then
        PT_FOUND="${PT_FILES[0]}"
    else
        echo "[ERROR] 发现 ${#PT_FILES[@]} 个 .pt 文件且无 best.pt"
        for f in "${PT_FILES[@]}"; do echo "  $f"; done
        exit 1
    fi
fi

# ---- 固定路径 ----
SAM3_OUT_DIR="${OUTPUT_DIR}/sam3_output"
JUDGE_OUT_DIR="${OUTPUT_DIR}/judge_output"

echo "============================================================"
echo "  Pipeline v2: ${FOLDER_NAME}"
echo "  Mode:       ${MODE}"
echo "  Input:      ${INPUT_DIR}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  GPU:        ${GPU_DEVICE:-all}"
if [ "$MODE" = "yolo" ]; then
    if [ "$MULTI_MODEL" -eq 1 ]; then
        echo "  YOLO:       multi-model (batch_01.py auto-match)"
    else
        echo "  YOLO model: ${PT_FOUND:-auto}"
    fi
fi
echo "============================================================"

eval "$(conda shell.bash hook)"

# ---- Step 1: SAM3 推理 ----
echo ""
echo "[Step 1/4] SAM3 推理 (${MODE} 模式)..."
conda activate "$SAM3_CONDA_ENV"

mkdir -p "$SAM3_OUT_DIR"

if [ "$MODE" = "dataset" ]; then
    python /home/model/work/sam3_facebook/batch_01.py \
        --dataset-root "$INPUT_DIR" \
        --splits $SPLITS \
        --output-dir "$SAM3_OUT_DIR" \
        --checkpoint "$SAM3_CHECKPOINT"

    # 把各 split 的 Instance/ 重命名为 Instance_{split}/
    INSTANCE_DIRS=()
    for split in $SPLITS; do
        src="${SAM3_OUT_DIR}/${FOLDER_NAME}/${split}/Instance"
        dst="${SAM3_OUT_DIR}/${FOLDER_NAME}/Instance_${split}"
        if [ -d "$src" ] && [ -f "${src}/instances_${split}.json" ]; then
            mv "$src" "$dst"
            INSTANCE_DIRS+=("$dst")
            echo "  生成: Instance_${split}/"
        fi
    done
else
    python /home/model/work/sam3_facebook/batch_01.py \
        --dataset-root "$INPUT_DIR" \
        --output-dir "$SAM3_OUT_DIR" \
        --checkpoint "$SAM3_CHECKPOINT"

    # Auto-detect Instance directories at maxdepth 2 under SAM3_OUT_DIR
    # Covers: sam3_output/{sub}/Instance/ (multi-model)
    #         sam3_output/{sub}/Instance_xxx/ (multi-model with prefix)
    #         sam3_output/{FOLDER}/Instance/ (single-model)
    #         sam3_output/{FOLDER}/Instance_xxx/ (single-model with prefix)
    INSTANCE_DIRS=()
    while IFS= read -r -d '' d; do
        INSTANCE_DIRS+=("$d")
    done < <(find "$SAM3_OUT_DIR" -maxdepth 3 -type d -name "Instance" -print0 2>/dev/null | sort -z)
    while IFS= read -r -d '' d; do
        INSTANCE_DIRS+=("$d")
    done < <(find "$SAM3_OUT_DIR" -maxdepth 3 -type d -name "Instance_*" -print0 2>/dev/null | sort -z)

    if [ ${#INSTANCE_DIRS[@]} -eq 0 ]; then
        echo "[ERROR] SAM3 未生成任何 Instance/ 目录"
        exit 1
    fi

    echo "[Step 1/4] 发现 ${#INSTANCE_DIRS[@]} 个 Instance 目录:"
    for d in "${INSTANCE_DIRS[@]}"; do echo "  - $(basename "$d")"; done
fi

# ---- Step 2-4: 对每个 Instance 循环执行 Judge → Report → Rerun ----
for INST_DIR in "${INSTANCE_DIRS[@]}"; do
    INST_NAME="$(basename "$INST_DIR")"
    SUB_COCO=$(find "${INST_DIR}" -maxdepth 1 -name "instances_*.json" -type f | head -1)

    if [ -z "$SUB_COCO" ] || [ ! -f "$SUB_COCO" ]; then
        echo "[WARN] 跳过 ${INST_NAME}: 未找到 instances_*.json"
        continue
    fi

    # 确定该 Instance 对应的图片目录和输出子目录
    # Multi-model: INST_DIR = sam3_output/{sub_name}/Instance/
    # Single-model Instance_*: INST_DIR = sam3_output/{FOLDER_NAME}/Instance_xxx/
    # Single-model Instance: INST_DIR = sam3_output/{FOLDER_NAME}/Instance/
    if [[ "$INST_NAME" == Instance_* ]]; then
        SUB_NAME="${INST_NAME#Instance_}"
        if [ "$MODE" = "dataset" ]; then
            SUB_IMG_DIR="${INPUT_DIR}/dataset/${SUB_NAME}/images"
        else
            SUB_IMG_DIR="${INPUT_DIR}/production_data/${SUB_NAME}"
        fi
        SUB_JUDGE_OUT="${JUDGE_OUT_DIR}/${SUB_NAME}"
    elif [ "$INST_NAME" = "Instance" ]; then
        PARENT_NAME="$(basename "$(dirname "$INST_DIR")")"
        if [ "$PARENT_NAME" != "$FOLDER_NAME" ]; then
            # Multi-model: parent is sub-material name
            SUB_NAME="$PARENT_NAME"
            SUB_IMG_DIR="${INPUT_DIR}/production_data/${SUB_NAME}"
            SUB_JUDGE_OUT="${JUDGE_OUT_DIR}/${SUB_NAME}"
        else
            # Single-model: parent is FOLDER_NAME
            SUB_NAME="${FOLDER_NAME}"
            if [ -d "${INPUT_DIR}/production_data" ]; then
                SUB_IMG_DIR="${INPUT_DIR}/production_data"
            else
                SUB_IMG_DIR="${INPUT_DIR}"
            fi
            SUB_JUDGE_OUT="${JUDGE_OUT_DIR}"
        fi
    else
        SUB_NAME="${FOLDER_NAME}"
        if [ -d "${INPUT_DIR}/production_data" ]; then
            SUB_IMG_DIR="${INPUT_DIR}/production_data"
        else
            SUB_IMG_DIR="${INPUT_DIR}"
        fi
        SUB_JUDGE_OUT="${JUDGE_OUT_DIR}"
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "  处理 Instance: ${INST_NAME} (${SUB_NAME})"
    echo "  COCO:   ${SUB_COCO}"
    echo "  Images: ${SUB_IMG_DIR}"
    echo "  Judge:  ${SUB_JUDGE_OUT}"
    echo "------------------------------------------------------------"

    # ---- Step 2: Qwen-VL 质检 ----
    echo ""
    echo "[Step 2/4] Qwen-VL 质检 (${SUB_NAME})..."
    conda activate "$JUDGE_CONDA_ENV"

    python /home/model/work/llm/judge_01.py \
        --images-dir "$SUB_IMG_DIR" \
        --coco-json "$SUB_COCO" \
        --output-dir "$SUB_JUDGE_OUT" \
        --model-path "$JUDGE_MODEL_PATH" \
        --max-images "$MAX_IMAGES" \
        --pass-threshold "$JUDGE_PASS_THRESHOLD"

    JUDGE_REPORT="${SUB_JUDGE_OUT}/judge_report.json"
    if [ ! -f "$JUDGE_REPORT" ]; then
        echo "[ERROR] judge 未生成报告: $JUDGE_REPORT"
        continue
    fi
    echo "[Step 2/4] 完成: $JUDGE_REPORT"

    # ---- Step 3: 结果汇总 ----
    echo ""
    echo "[Step 3/4] 结果汇总 (${SUB_NAME})..."
    python /home/model/work/llm/check_report_01.py "$JUDGE_REPORT"

    # ---- Step 4: 失败图片再优化 ----
    if [ "$SKIP_RERUN" = "1" ]; then
        echo ""
        echo "[Step 4/4] SKIP (SKIP_RERUN=1)"
    else
        echo ""
        echo "[Step 4/4] 失败图片再优化 (${SUB_NAME})..."
        conda activate "$JUDGE_CONDA_ENV"

        python /home/model/work/llm/rerun_optimize.py \
            --judge-report "$JUDGE_REPORT" \
            --coco-json "$SUB_COCO" \
            --images-dir "$SUB_IMG_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --model-path "$JUDGE_MODEL_PATH" \
            --sam3-checkpoint "$SAM3_CHECKPOINT" \
            --sam3-conda-env "$SAM3_CONDA_ENV" \
            --pass-threshold "$JUDGE_PASS_THRESHOLD"

        echo "[Step 4/4] 再优化完成"
    fi
done

echo ""
echo "============================================================"
echo "  Pipeline v2 完成: ${FOLDER_NAME}"
echo "  SAM3 output:  ${SAM3_OUT_DIR}"
echo "  Judge output: ${JUDGE_OUT_DIR}"
echo "============================================================"
