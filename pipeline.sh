#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 三阶段流水线: SAM3 推理 → Qwen-VL 质检 → 结果汇总
#
# 用法:
#   IMAGE_DIR=/path/to/images  \
#   YOLO_MODEL=/path/to/best.pt \
#   OUTPUT_BASE=/path/to/output \
#   bash pipeline.sh
#
# 必填环境变量:
#   IMAGE_DIR        原始图片目录
#   YOLO_MODEL       YOLO 模型权重路径
#   OUTPUT_BASE      输出根目录
#
# 可选环境变量:
#   SAM3_CHECKPOINT      SAM3 模型权重 (默认 /home/model/sam3_pth/sam3pt/sam3.pt)
#   JUDGE_MODEL_PATH      Qwen-VL 模型路径 (默认 /home/model/llm/qwen_4b)
#   MAX_IMAGES            judge 阶段最大图片数 (默认 0 = 全部)
#   JUDGE_PASS_THRESHOLD  合格分数阈值 (默认 0.75)
#   SAM3_CONDA_ENV        SAM3 conda 环境名 (默认 sam3_clone)
#   JUDGE_CONDA_ENV       judge conda 环境名 (默认 vllm_new)
# ============================================================

eval "$(conda shell.bash hook)"

IMAGE_DIR="${IMAGE_DIR:?请设置 IMAGE_DIR}"
OUTPUT_BASE="${OUTPUT_BASE:?请设置 OUTPUT_BASE}"
YOLO_MODEL="${YOLO_MODEL:?请设置 YOLO_MODEL (YOLO 模型权重路径)}"

SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/home/model/sam3_pth/sam3pt/sam3.pt}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/home/model/llm_model/qwen_4b}"
MAX_IMAGES="${MAX_IMAGES:-0}"
JUDGE_PASS_THRESHOLD="${JUDGE_PASS_THRESHOLD:-0.75}"
SAM3_CONDA_ENV="${SAM3_CONDA_ENV:-sam3_clone}"
JUDGE_CONDA_ENV="${JUDGE_CONDA_ENV:-vllm_new}"

SAM3_OUT_DIR="${OUTPUT_BASE}/sam3_output"
JUDGE_OUT_DIR="${OUTPUT_BASE}/judge_output"

echo "============================================================"
echo "  Pipeline: SAM3 → Judge → Check"
echo "  Images:      $IMAGE_DIR"
echo "  Output base: $OUTPUT_BASE"
echo "============================================================"

# ---- Step 1: SAM3 推理 ----
echo ""
echo "[Step 1/3] SAM3 推理..."
conda activate "$SAM3_CONDA_ENV"
python /home/model/work/sam3_facebook/batch.py \
    --image-folder "$IMAGE_DIR" \
    --yolo-model "$YOLO_MODEL" \
    --output-dir "$SAM3_OUT_DIR" \
    --checkpoint "$SAM3_CHECKPOINT"

COCO_JSON="${SAM3_OUT_DIR}/Instance/instances_default.json"
if [ ! -f "$COCO_JSON" ]; then
    echo "[ERROR] SAM3 未生成 COCO JSON: $COCO_JSON"
    exit 1
fi
echo "[Step 1/3] 完成: $COCO_JSON"

# ---- Step 2: Qwen-VL 质检 ----
echo ""
echo "[Step 2/3] Qwen-VL 质检..."
conda activate "$JUDGE_CONDA_ENV"
python /home/model/work/llm/judge.py \
    --images-dir "$IMAGE_DIR" \
    --coco-json "$COCO_JSON" \
    --output-dir "$JUDGE_OUT_DIR" \
    --model-path "$JUDGE_MODEL_PATH" \
    --max-images "$MAX_IMAGES" \
    --pass-threshold "$JUDGE_PASS_THRESHOLD"

JUDGE_REPORT="${JUDGE_OUT_DIR}/judge_report.json"
if [ ! -f "$JUDGE_REPORT" ]; then
    echo "[ERROR] judge 未生成报告: $JUDGE_REPORT"
    exit 1
fi
echo "[Step 2/3] 完成: $JUDGE_REPORT"

# ---- Step 3: 结果汇总 ----
echo ""
echo "[Step 3/3] 结果汇总..."
python /home/model/work/llm/check_report.py "$JUDGE_REPORT"
echo ""
echo "============================================================"
echo "  Pipeline 完成"
echo "  SAM3 output:  $SAM3_OUT_DIR"
echo "  Judge output: $JUDGE_OUT_DIR"
echo "============================================================"
