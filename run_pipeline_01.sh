#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 自动探测 pipeline: dataset 模式 / YOLO 模式 → SAM3 → Judge → Report
#
# 用法:
#   bash run_pipeline.sh <input_dir> [output_dir]
#
#   input_dir:  如 /root/piplinetest/辣椒包堆叠/
#   output_dir: 可选, 默认 {input_dir}/pipeline_output/
#
# 可选环境变量:
#   SAM3_CHECKPOINT      SAM3 模型权重 (默认 /home/model/sam3_pth/sam3pt/sam3.pt)
#   JUDGE_MODEL_PATH      Qwen-VL 模型路径 (默认 /home/model/llm_model/qwen_4b)
#   MAX_IMAGES            judge 阶段最大图片数 (默认 0 = 全部)
#   JUDGE_PASS_THRESHOLD  合格分数阈值 (默认 0.75)
#   SAM3_CONDA_ENV        SAM3 conda 环境名 (默认 sam3_6000d)
#   JUDGE_CONDA_ENV       judge conda 环境名 (默认 vllm_new)
#   GPU_DEVICE            指定 GPU 卡号 (如 0 或 1), 不设则使用全部 GPU
# ============================================================

INPUT_DIR="${1:?用法: bash run_pipeline.sh <input_dir> [output_dir]}"

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

# ---- GPU 绑定: 所有 Python 子进程仅可见指定 GPU ----
if [ -n "$GPU_DEVICE" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
fi

# ---- Step 0: 探测模式 ----
if [ -d "${INPUT_DIR}/dataset" ]; then
    MODE="dataset"
    SPLITS="train valid test"
    JUDGE_IMAGE_DIR="${INPUT_DIR}/dataset"
else
    MODE="yolo"
    # 找 .pt 文件: 优先 best.pt, 否则只用唯一的 .pt, >1 个且无 best.pt 则跳过
    PT_FILES=()
    PT_FOUND=""
    while IFS= read -r -d '' f; do
        PT_FILES+=("$f")
    done < <(find "${INPUT_DIR}" -maxdepth 3 -name "*.pt" -type f -print0 2>/dev/null || true)

    if [ "${#PT_FILES[@]}" -eq 0 ]; then
        echo "[ERROR] 未找到 .pt 模型文件"
        exit 1
    fi

    # 统计 best.pt 数量（只允许 1 个，多个 best.pt 无法确定用哪个→跳过）
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
        echo "[ERROR] 发现 ${BEST_PT_COUNT} 个 best.pt 文件，无法确定用哪个，跳过此文件夹"
        for f in "${PT_FILES[@]}"; do echo "  $f"; done
        exit 1
    elif [ "${#PT_FILES[@]}" -eq 1 ]; then
        PT_FOUND="${PT_FILES[0]}"
    else
        echo "[ERROR] 发现 ${#PT_FILES[@]} 个 .pt 文件且无 best.pt，跳过此文件夹"
        for f in "${PT_FILES[@]}"; do echo "  $f"; done
        exit 1
    fi
fi

# ---- 固定路径 ----
SAM3_OUT_DIR="${OUTPUT_DIR}/sam3_output"
JUDGE_OUT_DIR="${OUTPUT_DIR}/judge_output"

echo "============================================================"
echo "  Pipeline: ${FOLDER_NAME}"
echo "  Mode:       ${MODE}"
echo "  Input:      ${INPUT_DIR}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  GPU:        ${GPU_DEVICE:-all}"
if [ "$MODE" = "yolo" ]; then
    echo "  YOLO model: ${PT_FOUND}"
fi
echo "============================================================"

eval "$(conda shell.bash hook)"

# ---- Step 1: SAM3 推理 ----
echo ""
echo "[Step 1/3] SAM3 推理 (${MODE} 模式)..."
conda activate "$SAM3_CONDA_ENV"

mkdir -p "$SAM3_OUT_DIR"

if [ "$MODE" = "dataset" ]; then
    python /home/model/work/sam3_facebook/batch_01.py \
        --dataset-root "$INPUT_DIR" \
        --splits $SPLITS \
        --output-dir "$SAM3_OUT_DIR" \
        --checkpoint "$SAM3_CHECKPOINT"

    # 合并 train/valid/test 的 COCO JSON → instances_default.json
    MERGED_DIR="${SAM3_OUT_DIR}/${FOLDER_NAME}/Instance"
    mkdir -p "$MERGED_DIR"

    python -c "
import json
from pathlib import Path

merged = {'images': [], 'annotations': [], 'categories': []}
img_offset = 1
ann_offset = 1

for split in ['train', 'valid', 'test']:
    p = Path('${SAM3_OUT_DIR}/${FOLDER_NAME}') / split / 'Instance' / f'instances_{split}.json'
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    id_map = {}
    for img in d.get('images', []):
        new_id = img_offset
        id_map[img['id']] = new_id
        img['id'] = new_id
        img_offset += 1
        merged['images'].append(img)
    for ann in d.get('annotations', []):
        ann['id'] = ann_offset
        ann['image_id'] = id_map.get(ann['image_id'], ann['image_id'])
        ann_offset += 1
        merged['annotations'].append(ann)
    for cat in d.get('categories', []):
        if not any(c['id'] == cat['id'] for c in merged['categories']):
            merged['categories'].append(cat)

out = Path('${MERGED_DIR}/instances_default.json')
out.write_text(json.dumps(merged, indent=2), encoding='utf-8')
print(f'合并完成: {out} ({len(merged[\"images\"])} images, {len(merged[\"annotations\"])} annots)')
"

    COCO_JSON="${MERGED_DIR}/instances_default.json"
else
    python /home/model/work/sam3_facebook/batch_01.py \
        --dataset-root "$INPUT_DIR" \
        --output-dir "$SAM3_OUT_DIR" \
        --checkpoint "$SAM3_CHECKPOINT"

    COCO_JSON="${SAM3_OUT_DIR}/${FOLDER_NAME}/Instance/instances_default.json"
fi

if [ ! -f "$COCO_JSON" ]; then
    echo "[ERROR] SAM3 未生成 COCO JSON: $COCO_JSON"
    exit 1
fi
echo "[Step 1/3] 完成: $COCO_JSON"

# ---- Step 2: Qwen-VL 质检 ----
echo ""
echo "[Step 2/3] Qwen-VL 质检..."
conda activate "$JUDGE_CONDA_ENV"

if [ "$MODE" = "yolo" ]; then
    JUDGE_IMAGE_DIR="${INPUT_DIR}/production_data"
    if [ ! -d "$JUDGE_IMAGE_DIR" ]; then
        JUDGE_IMAGE_DIR="$INPUT_DIR"
    fi
fi

python /home/model/work/llm/judge_01.py \
    --images-dir "$JUDGE_IMAGE_DIR" \
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
python /home/model/work/llm/check_report_01.py "$JUDGE_REPORT"

echo ""
echo "============================================================"
echo "  Pipeline 完成: ${FOLDER_NAME}"
echo "  SAM3 output:  ${SAM3_OUT_DIR}"
echo "  Judge output: ${JUDGE_OUT_DIR}"
echo "============================================================"
