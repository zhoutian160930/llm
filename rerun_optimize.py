#!/usr/bin/env python3
"""
rerun_optimize.py — 失败图片再优化脚本

流程:
  1. 读取 judge_report.json, 找到 fail + review 的图片
  2. 用 Qwen 分析每个 bbox 区域, 生成文本描述
  3. 调用 SAM3 混合提示 (box + text) 重新分割
  4. 重新 judge 评估
  5. 合并真值: instances_all.json + instances_pass_only.json

运行环境: vllm_new (conda)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from tqdm import tqdm
from vllm import LLM, SamplingParams

# vLLM 环境变量
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# ============ Qwen Text Description Functions ============

DESCRIBE_SYSTEM_PROMPT = """你是一个物体描述专家。你将看到一张从图片中裁剪出来的物体区域。
请用一句简洁的中文描述这个物体的外观特征，包括颜色、形状、材质、纹理、姿态等信息。
只输出描述本身，不要加任何前缀、解释或标点之外的符号。描述控制在20字以内。"""


def prepare_vllm_inputs(messages, processor):
    """Prepare inputs for vLLM inference (reused from judge_01.py pattern)."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {
        "prompt": text,
        "multi_modal_data": mm_data,
    }


def qwen_describe_object(llm, processor, sampling_params, crop_image_path):
    """Send a cropped image to Qwen and get a short text description."""
    messages = [
        {"role": "system", "content": DESCRIBE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop_image_path},
                {"type": "text", "text": "请描述这个物体。"},
            ],
        },
    ]
    inputs = prepare_vllm_inputs(messages, processor)
    outputs = llm.generate([inputs], sampling_params=sampling_params)
    text = outputs[0].outputs[0].text.strip()
    # Clean up: remove quotes, extra punctuation at ends
    text = text.strip('"\'」「').strip()
    return text


# ============ Judge Functions (simplified from judge_01.py) ============

def load_coco_fn(coco_path):
    with open(coco_path, 'r') as f:
        return json.load(f)


def extract_json_block(text):
    """Extract JSON from model output, handling optional code fences."""
    text = text.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Find first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def clamp_score(value):
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return 0.0


def normalize_judge_result(raw, anns, pass_threshold):
    """
    Normalize judge output to match annotation IDs.
    Handles both 'items' (standard) and 'instance_results' (Qwen-4B variant) formats.
    """
    # Support both output formats
    items = raw.get("items") or raw.get("instance_results") or []

    # Map quality_status to standard status
    STATUS_MAP = {
        "good": "good", "acceptable": "good", "excellent": "good", "accurate": "good",
        "bad": "bad", "poor": "bad", "incorrect": "bad",
        "uncertain": "uncertain", "review": "uncertain",
    }

    ann_by_id = {int(a.get("id", -1)): a for a in anns}
    ann_by_viz = {int(a.get("viz_index", -1)): a for a in anns}

    normalized_items = []
    all_scores = []

    for item in items:
        aid = int(item.get("annotation_id", -1))
        viz = int(item.get("viz_index", -1))
        matched = ann_by_id.get(aid) or ann_by_viz.get(viz)
        if matched is None:
            continue

        # Score: handle both 0-1 and 0-100 scales
        score_raw = item.get("score") or item.get("total_score") or 0
        score = clamp_score(score_raw / 100.0 if isinstance(score_raw, (int, float)) and score_raw > 1 else score_raw)

        # Status: handle quality_status, status, or infer from score
        status_raw = str(item.get("status") or item.get("quality_status") or "").lower().strip()
        status = STATUS_MAP.get(status_raw, "")
        if not status:
            if score >= pass_threshold:
                status = "good"
            elif score < 0.4:
                status = "bad"
            else:
                status = "uncertain"

        normalized_items.append({
            "annotation_id": int(matched.get("id", -1)),
            "viz_index": int(matched.get("viz_index", -1)),
            "status": status,
            "score": score,
            "reason": str(item.get("reason", "")).strip(),
            "suggestion": str(item.get("suggestion", "")).strip(),
            "category_id": int(matched.get("category_id", 1)),
        })
        all_scores.append(score)

    # Fill missing annotations as uncertain
    seen_aids = {x["annotation_id"] for x in normalized_items}
    for a in anns:
        aid = int(a.get("id", -1))
        if aid not in seen_aids:
            normalized_items.append({
                "annotation_id": aid,
                "viz_index": int(a.get("viz_index", -1)),
                "status": "uncertain",
                "score": 0.5,
                "reason": "模型未对该实例输出评估",
                "suggestion": "",
                "category_id": int(a.get("category_id", 1)),
            })
            all_scores.append(0.5)

    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    statuses = {x["status"] for x in normalized_items}
    if "bad" in statuses:
        overall_status = "fail"
    elif "uncertain" in statuses:
        overall_status = "review"
    elif overall_score >= pass_threshold:
        overall_status = "pass"
    else:
        overall_status = "review"

    return {
        "overall_status": overall_status,
        "overall_score": round(overall_score, 4),
        "summary": str(raw.get("summary", "")).strip(),
        "items": normalized_items,
    }


def find_image_file(images_dir, file_name):
    """Find image by file_name under images_dir (recursive)."""
    images_dir = Path(images_dir)
    direct = images_dir / file_name
    if direct.exists():
        return direct
    for p in images_dir.rglob(file_name):
        return p
    return None


# Color palette for overlay (matches judge_01.py)
PALETTE = (
    (239, 83, 80), (102, 187, 106), (66, 165, 245), (255, 202, 40),
    (171, 71, 188), (255, 112, 67), (38, 198, 218), (156, 204, 101),
)


def color_for_index(idx):
    return PALETTE[idx % len(PALETTE)]


def _polygon_list_from_segmentation(segmentation):
    """Extract polygon list from COCO segmentation."""
    if not segmentation:
        return []
    if isinstance(segmentation, list):
        if all(isinstance(x, (int, float)) for x in segmentation):
            return [segmentation]
        return segmentation
    return []



def ann_to_mask(ann, width, height):
    """Convert annotation polygons to binary mask."""
    polys = _polygon_list_from_segmentation(ann.get("segmentation", []))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polys:
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return mask


def build_overlay_fn(image, anns, render_width=None, render_height=None):
    """Build overlay image from annotations (simplified from judge_01.py)."""
    if render_width and render_height:
        overlay = image.copy().resize((render_width, render_height), Image.LANCZOS)
    else:
        overlay = image.copy()
    overlay_np = np.array(overlay, dtype=np.float32)

    scale_x = render_width / image.width if (render_width and image.width) else 1.0
    scale_y = render_height / image.height if (render_height and image.height) else 1.0

    for idx, ann in enumerate(anns):
        color = color_for_index(idx)
        mask = ann_to_mask(ann, render_width or image.width, render_height or image.height)
        mask_np = np.array(mask, dtype=bool)
        overlay_np[mask_np] = (1 - 0.5) * overlay_np[mask_np] + 0.5 * np.array(color)

        bbox = ann.get("bbox", [0, 0, 0, 0])
        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int((bbox[0] + bbox[2]) * scale_x)
        y2 = int((bbox[1] + bbox[3]) * scale_y)

        draw = ImageDraw.Draw(overlay)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except Exception:
            font = None
        draw.text((x1 + 2, y1 + 2), f"#{idx}", fill=color, font=font)

    return Image.fromarray(overlay_np.astype(np.uint8))


def resize_for_upload_fn(image, max_side):
    """Resize image for model input."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def make_simple_user_prompt(image_name, width, height, categories, anns):
    """Build a simple user prompt for judge."""
    cat_lines = "\n".join([f"  - id {cid}: {name}" for cid, name in sorted(categories.items())])
    ann_lines = []
    for a in anns:
        vidx = a.get("viz_index", "?")
        aid = a.get("id", "?")
        cid = a.get("category_id", "?")
        bbox = a.get("bbox", [0, 0, 0, 0])
        ann_lines.append(f"  - viz_index {vidx}, annotation_id {aid}, category_id {cid}, bbox {bbox}")
    annotations_text = "\n".join(ann_lines) if ann_lines else "  (none)"

    return f"""现在开始判断样本: {image_name}
图像尺寸: {width} x {height}
目标类别列表:
{cat_lines}

每个实例的元数据:
{annotations_text}

请按照系统提示中的标准逐一对每个实例进行评分，并以纯 JSON 格式输出结果。"""


def judge_single_image(llm, processor, sampling_params, image_path, anns,
                       system_prompt, pass_threshold, input_dir, image_id, file_name, orig_width, orig_height):
    """
    Judge a single image and return (overall_status, overall_score, per_image_dict).
    per_image_dict matches judge_01.py output format.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Add viz_index to annotations
    for idx, a in enumerate(sorted(anns, key=lambda x: int(x.get("id", 0)))):
        a["viz_index"] = idx

    resized = resize_for_upload_fn(img, 1600)
    overlay = build_overlay_fn(resized, anns, render_width=w, render_height=h)

    # Build categories map
    categories = defaultdict(str)
    for a in anns:
        cid = int(a.get("category_id", 1))
        categories[cid] = f"class_{cid}"

    user_prompt = make_simple_user_prompt(
        file_name, resized.width, resized.height, categories, anns
    )

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "第一张是原图，第二张是分割可视化图。请输出JSON格式评估结果。"},
                {"type": "image", "image": str(image_path)},
                {"type": "image", "image": str(image_path)},  # placeholder for overlay
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    # Save resized + overlay to temp
    tmp_dir = Path(input_dir) / ".rerun_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    resized_path = tmp_dir / f"resized_{file_name}"
    overlay_path = tmp_dir / f"overlay_{file_name}"
    resized.save(str(resized_path), quality=92)
    overlay.save(str(overlay_path), quality=92)

    messages[1]["content"][1]["image"] = str(resized_path)
    messages[1]["content"][2]["image"] = str(overlay_path)

    inputs_data = prepare_vllm_inputs(messages, processor)
    outputs = llm.generate([inputs_data], sampling_params=sampling_params)
    raw_text = outputs[0].outputs[0].text.strip()

    parsed = extract_json_block(raw_text)
    judged = normalize_judge_result(parsed, anns=anns, pass_threshold=pass_threshold)

    # Build per-image output matching judge_01.py format
    per_image_output = {
        "image": {
            "id": image_id,
            "file_name": file_name,
            "path": str(image_path),
            "width": orig_width,
            "height": orig_height,
            "model_input_path": str(resized_path),
            "model_input_width": resized.width,
            "model_input_height": resized.height,
        },
        "judge": judged,
        "prompt": {
            "system": system_prompt,
            "user": user_prompt,
            "fewshot_examples": None,
        },
        "raw_model_output": raw_text,
    }

    # Keep temp files for reference
    return judged["overall_status"], judged["overall_score"], per_image_output


# ============ COCO Merge ============

def merge_coco_jsons(original_coco, rerun_coco, image_paths_passed_rerun):
    """
    Merge original COCO with rerun COCO.
    - instances_all.json: original + rerun (replace original entries for rerun images)
    - instances_pass_only.json: only images that passed (original pass + rerun pass)
    """
    orig = load_coco_fn(original_coco) if isinstance(original_coco, (str, Path)) else original_coco
    rerun = load_coco_fn(rerun_coco) if isinstance(rerun_coco, (str, Path)) else rerun_coco

    orig_images = {img["file_name"]: img for img in orig.get("images", [])}
    rerun_images = {img["file_name"]: img for img in rerun.get("images", [])}

    # Build annotation lookup by image_id
    orig_anns_by_img_id = defaultdict(list)
    for ann in orig.get("annotations", []):
        orig_anns_by_img_id[int(ann["image_id"])].append(ann)

    # Categories: merge, keeping original IDs
    categories = list(orig.get("categories", []))
    cat_ids = {c["id"] for c in categories}

    # Build all-inclusive
    all_images = []
    all_annotations = []
    ann_id = 1
    img_id = 1

    # Process original images
    for img in orig.get("images", []):
        fname = img["file_name"]
        if fname in rerun_images:
            # Use rerun annotations (replace)
            rerun_img = rerun_images[fname]
            all_images.append({"id": img_id, "file_name": fname,
                               "width": rerun_img["width"], "height": rerun_img["height"]})
            for ann in rerun.get("annotations", []):
                if int(ann["image_id"]) == int(rerun_img["id"]):
                    all_annotations.append({**ann, "id": ann_id, "image_id": img_id})
                    ann_id += 1
            img_id += 1
        else:
            all_images.append({"id": img_id, "file_name": fname,
                               "width": img["width"], "height": img["height"]})
            for ann in orig_anns_by_img_id[int(img["id"])]:
                all_annotations.append({**ann, "id": ann_id, "image_id": img_id})
                ann_id += 1
            img_id += 1

    # Add rerun-only images (shouldn't happen, but handle)
    for fname, rimg in rerun_images.items():
        if fname not in orig_images:
            all_images.append({"id": img_id, "file_name": fname,
                               "width": rimg["width"], "height": rimg["height"]})
            for ann in rerun.get("annotations", []):
                if int(ann["image_id"]) == int(rimg["id"]):
                    all_annotations.append({**ann, "id": ann_id, "image_id": img_id})
                    ann_id += 1
            img_id += 1

    # Ensure all annotation category_ids exist in categories
    used_cat_ids = set()
    for ann in all_annotations:
        used_cat_ids.add(ann.get("category_id", 1))
    for cid in used_cat_ids:
        if cid not in cat_ids:
            categories.append({"id": cid, "name": f"class_{cid}"})

    coco_all = {"images": all_images, "annotations": all_annotations, "categories": categories}

    # Build pass-only
    pass_only_images = []
    pass_only_annotations = []
    ann_id = 1
    img_id = 1
    passed_fnames = set(image_paths_passed_rerun) if image_paths_passed_rerun else set()

    for fname in passed_fnames:
        if fname in rerun_images:
            rimg = rerun_images[fname]
            pass_only_images.append({"id": img_id, "file_name": fname,
                                     "width": rimg["width"], "height": rimg["height"]})
            for ann in rerun.get("annotations", []):
                if int(ann["image_id"]) == int(rimg["id"]):
                    pass_only_annotations.append({**ann, "id": ann_id, "image_id": img_id})
                    ann_id += 1
            img_id += 1
        elif fname in orig_images:
            oimg = orig_images[fname]
            pass_only_images.append({"id": img_id, "file_name": fname,
                                     "width": oimg["width"], "height": oimg["height"]})
            for ann in orig_anns_by_img_id.get(int(oimg["id"]), []):
                pass_only_annotations.append({**ann, "id": ann_id, "image_id": img_id})
                ann_id += 1
            img_id += 1

    coco_pass = {"images": pass_only_images, "annotations": pass_only_annotations, "categories": categories}

    return coco_all, coco_pass


# ============ Helper: Copy + Merge ============

def copy_dir_contents(src_dir, dst_dir):
    """Copy all files from src_dir into dst_dir (overwrite)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        for f in src_dir.iterdir():
            shutil.copy2(str(f), str(dst_dir / f.name))


def merge_judge_reports(original_report, rerun_report):
    """Replace entries in original report with rerun results for matching file_names."""
    rerun_by_file = {img["file_name"]: img for img in rerun_report.get("images", [])}

    updated_images = []
    pass_count = review_count = fail_count = 0
    total_score = 0.0

    for img in original_report.get("images", []):
        fname = img["file_name"]
        if fname in rerun_by_file:
            rerun_img = rerun_by_file[fname]
            img = dict(img)  # shallow copy
            img["overall_status"] = rerun_img["overall_status"]
            img["overall_score"] = rerun_img["overall_score"]
            img["num_items"] = rerun_img.get("num_items", img.get("num_items", 0))

        updated_images.append(img)
        total_score += img.get("overall_score", 0)
        status = img.get("overall_status", "review")
        if status == "pass":
            pass_count += 1
        elif status == "fail":
            fail_count += 1
        else:
            review_count += 1

    n = len(updated_images)
    original_report["images"] = updated_images
    original_report["summary"]["pass"] = pass_count
    original_report["summary"]["review"] = review_count
    original_report["summary"]["fail"] = fail_count
    original_report["summary"]["mean_overall_score"] = round(total_score / n, 4) if n > 0 else 0.0

    return original_report


def write_summary_txt(output_dir, failed_images, rerun_results, sam3_coco_path, judge_report_path):
    """Write rerun_summary.txt — which images re-run, which passed, where saved."""
    lines = []
    lines.append("Rerun Summary")
    lines.append("=" * 60)
    lines.append(f"Total images re-run: {len(rerun_results)}")

    pass_after = sum(1 for v in rerun_results.values() if v["status"] == "pass")
    still_fail = len(rerun_results) - pass_after
    lines.append(f"Pass after re-run: {pass_after}")
    lines.append(f"Still fail/review: {still_fail}")
    lines.append("")
    lines.append("Re-run image details:")
    lines.append("-" * 60)

    for img_info in failed_images:
        fname = img_info["file_name"]
        rr = rerun_results.get(fname)
        if rr:
            orig_status = img_info.get("overall_status", "?")
            new_status = rr["status"]
            score = rr["score"]
            lines.append(f"  {fname}: {orig_status} -> {new_status} (score: {score:.4f})")

    lines.append("")
    lines.append("-" * 60)
    lines.append("Output locations:")
    lines.append(f"  SAM3 merged COCO: {sam3_coco_path}")
    lines.append(f"  Judge report:     {judge_report_path}")

    # Pass-only image list
    lines.append("")
    lines.append("Pass images after re-run:")
    for img_info in failed_images:
        fname = img_info["file_name"]
        rr = rerun_results.get(fname)
        if rr and rr["status"] == "pass":
            lines.append(f"  [RE-RUN PASS] {fname}")

    txt_path = output_dir / "rerun_summary.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[INFO] Summary written to: {txt_path}")


# ============ Main Orchestrator ============

def parse_args():
    parser = argparse.ArgumentParser(description="失败图片再优化 — Qwen描述 + SAM3混合提示 + 重新评判")
    parser.add_argument("--judge-report", type=Path, required=True,
                        help="judge_report.json 路径")
    parser.add_argument("--coco-json", type=Path, required=True,
                        help="原始 COCO JSON (用于获取 bbox)")
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="图片根目录")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="输出目录 (将在此创建 rerun_*)")
    parser.add_argument("--model-path", type=str,
                        default="/home/model/llm_model/qwen_4b",
                        help="Qwen-VL 模型路径")
    parser.add_argument("--sam3-checkpoint", type=str,
                        default="/home/model/sam3_pth/sam3pt/sam3.pt",
                        help="SAM3 checkpoint")
    parser.add_argument("--sam3-conda-env", type=str, default="sam3_6000d",
                        help="SAM3 conda 环境名")
    parser.add_argument("--pass-threshold", type=float, default=0.75,
                        help="合格分数阈值")
    parser.add_argument("--max-images", type=int, default=0,
                        help="最大重跑图片数 (0=全部)")
    parser.add_argument("--skip-describe", action="store_true",
                        help="跳过 Qwen 描述步骤 (使用默认 text_prompt)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    tmp_dir = output_dir / ".rerun_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sam3_instance_dir = args.coco_json.parent
    judge_output_dir = args.judge_report.parent

    # ---- 1. Read judge report, find fail+review images ----
    with open(args.judge_report, 'r') as f:
        report = json.load(f)

    failed_images = [img for img in report.get("images", [])
                     if img.get("overall_status") in ("fail", "review")]

    if not failed_images:
        print("[INFO] No fail/review images to rerun")
        failed_names = []
        rerun_results = {}
        write_summary_txt(output_dir, [], rerun_results, args.coco_json, args.judge_report)
        return

    if args.max_images > 0:
        failed_images = failed_images[:args.max_images]

    print(f"[INFO] Found {len(failed_images)} fail/review images")

    coco_data = load_coco_fn(args.coco_json)
    coco_anns_by_img = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        coco_anns_by_img[int(ann["image_id"])].append(ann)
    file_to_id = {img["file_name"]: img["id"] for img in coco_data.get("images", [])}

    # ---- 2. Load Qwen for global description ----
    gpu_count = torch.cuda.device_count()
    tensor_parallel_size = min(1, gpu_count)
    print(f"[INFO] Loading Qwen model: {args.model_path} (GPU count: {gpu_count})")

    processor = AutoProcessor.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,
        tensor_parallel_size=tensor_parallel_size,
        seed=0,
        max_model_len=16384,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        disable_log_stats=True,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=256, top_k=-1, stop_token_ids=[])

    # ---- 3. Global text description (one Qwen call for all failed images) ----
    prompt_input = {"images": []}
    crop_dir = tmp_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    global_text_desc = "object"
    if not args.skip_describe:
        first_img_info = failed_images[0]
        first_file = first_img_info["file_name"]
        first_iid = file_to_id.get(first_file)
        if first_iid is not None:
            first_img_path = find_image_file(args.images_dir, first_file)
            first_anns = coco_anns_by_img.get(int(first_iid), [])
            if first_img_path and first_anns:
                try:
                    pil_img = Image.open(first_img_path).convert("RGB")
                    w, h = pil_img.size
                    first_bbox = first_anns[0].get("bbox", [0, 0, w, h])
                    fx, fy, fbw, fbh = first_bbox
                    margin_x = fbw * 0.1
                    margin_y = fbh * 0.1
                    x1 = max(0, int(fx - margin_x))
                    y1 = max(0, int(fy - margin_y))
                    x2 = min(w, int(fx + fbw + margin_x))
                    y2 = min(h, int(fy + fbh + margin_y))
                    crop = pil_img.crop((x1, y1, x2, y2))
                    crop_path = crop_dir / f"{first_file.rsplit('.',1)[0]}_ref.jpg"
                    crop.save(str(crop_path), quality=92)
                    try:
                        global_text_desc = qwen_describe_object(llm, processor, sampling_params, str(crop_path))
                    except Exception as e:
                        print(f"    [WARN] Qwen describe failed: {e}")
                    print(f"    [Global] desc='{global_text_desc}'")
                except Exception as e:
                    print(f"    [WARN] Cannot generate global desc: {e}")

    # ---- 4. Build SAM3 prompts ----
    for img_info in tqdm(failed_images, desc="Building prompts"):
        file_name = img_info["file_name"]
        image_id = file_to_id.get(file_name)
        if image_id is None:
            print(f"  [WARN] {file_name} not found in COCO, skipping")
            continue

        img_path = find_image_file(args.images_dir, file_name)
        if img_path is None:
            print(f"  [WARN] Image not found: {file_name}")
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Cannot open {img_path}: {e}")
            continue

        w, h = pil_img.size
        anns = coco_anns_by_img.get(int(image_id), [])
        if not anns:
            continue

        image_prompts = []
        for ann in anns:
            image_prompts.append({
                "bbox": ann.get("bbox", [0, 0, w, h]),
                "text": global_text_desc,
            })

        prompt_input["images"].append({
            "image_path": str(img_path),
            "width": w,
            "height": h,
            "prompts": image_prompts,
        })

    if not prompt_input["images"]:
        print("[ERROR] No valid images with prompts generated")
        sys.exit(1)

    prompt_json_path = tmp_dir / "sam3_prompts.json"
    prompt_json_path.write_text(json.dumps(prompt_input, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved {len(prompt_input['images'])} images with prompts to {prompt_json_path}")

    # Release Qwen GPU memory for SAM3
    print(f"\n[INFO] Releasing Qwen GPU memory...")
    del llm
    del processor
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[INFO] GPU memory freed: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB allocated")

    # ---- 5. Run SAM3 mixed-prompt re-segmentation ----
    sam3_tmp_dir = tmp_dir / "sam3_rerun_output"
    sam3_tmp_dir.mkdir(parents=True, exist_ok=True)

    sam3_rerun_script = "/home/model/work/sam3_facebook/sam3_mixed_rerun.py"
    print(f"\n[INFO] Running SAM3 mixed-prompt re-segmentation...")
    print(f"  Input:  {prompt_json_path}")
    print(f"  Output: {sam3_tmp_dir}")

    cmd = [
        "conda", "run", "-n", args.sam3_conda_env, "--no-capture-output",
        "python", sam3_rerun_script,
        "--input-json", str(prompt_json_path),
        "--output-dir", str(sam3_tmp_dir),
        "--checkpoint", args.sam3_checkpoint,
        "--batch-size", "1",
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] SAM3 rerun failed with exit code {result.returncode}")
        sys.exit(1)

    rerun_coco_path = sam3_tmp_dir / "instances_default.json"
    if not rerun_coco_path.exists():
        print(f"[ERROR] SAM3 rerun did not produce {rerun_coco_path}")
        sys.exit(1)

    print(f"[INFO] SAM3 rerun complete: {rerun_coco_path}")

    # ---- 6. Copy SAM3 per-image outputs into original sam3_output ----
    sam3_tmp_instance = sam3_tmp_dir / "Instance"
    print(f"\n[INFO] Copying SAM3 rerun outputs into {sam3_instance_dir} ...")
    for sub in ["label", "mask", "vis"]:
        src = sam3_tmp_instance / sub
        dst = sam3_instance_dir / sub
        if src.exists():
            n = sum(1 for _ in src.iterdir())
            copy_dir_contents(src, dst)
            print(f"  Copied {n} files: {sub}/")

    # ---- 7. Merge COCO → overwrite original instances_default.json ----
    print(f"\n[INFO] Merging COCO → overwriting {args.coco_json} ...")
    coco_all, _ = merge_coco_jsons(args.coco_json, rerun_coco_path, set())
    args.coco_json.write_text(json.dumps(coco_all, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Done: {len(coco_all['images'])} images, {len(coco_all['annotations'])} annotations")

    # ---- 8. Re-judge rerun images ----
    judge_tmp_dir = tmp_dir / "judge_output"
    judge_tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Re-judging rerun images (calling judge_01.py)...")
    judge_cmd = [
        "python", "/home/model/work/llm/judge_01.py",
        "--images-dir", str(args.images_dir),
        "--coco-json", str(rerun_coco_path),
        "--output-dir", str(judge_tmp_dir),
        "--model-path", args.model_path,
        "--pass-threshold", str(args.pass_threshold),
        "--max-images", "0",
    ]
    result = subprocess.run(judge_cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Re-judge failed with exit code {result.returncode}")
        sys.exit(1)

    rerun_judge_report_path = judge_tmp_dir / "judge_report.json"
    if not rerun_judge_report_path.exists():
        print(f"[ERROR] Re-judge did not produce {rerun_judge_report_path}")
        sys.exit(1)

    print(f"[INFO] Re-judge complete: {rerun_judge_report_path}")

    # ---- 9. Copy judge outputs into original judge_output ----
    print(f"\n[INFO] Copying re-judge outputs into {judge_output_dir} ...")
    for sub in ["json", "vis"]:
        src = judge_tmp_dir / sub
        dst = judge_output_dir / sub
        if src.exists():
            n = sum(1 for _ in src.iterdir())
            copy_dir_contents(src, dst)
            print(f"  Copied {n} files: {sub}/")

    # ---- 10. Parse re-judge results & update judge_report.json ----
    with open(rerun_judge_report_path, 'r') as f:
        rerun_judge_report = json.load(f)

    rerun_results = {}
    for img in rerun_judge_report.get("images", []):
        fname = img["file_name"]
        rerun_results[fname] = {
            "status": img.get("overall_status", "review"),
            "score": img.get("overall_score", 0.0),
        }
        print(f"  [{rerun_results[fname]['status']}] {fname} score={rerun_results[fname]['score']:.4f}")

    updated_report = merge_judge_reports(report, rerun_judge_report)
    args.judge_report.write_text(json.dumps(updated_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Updated judge report: {args.judge_report}")

    # ---- 11. Write summary txt ----
    write_summary_txt(output_dir, failed_images, rerun_results, args.coco_json, args.judge_report)

    print(f"\n{'='*60}")
    print(f"[DONE] Rerun complete")
    print(f"  SAM3 COCO:  {args.coco_json}")
    print(f"  Judge report: {args.judge_report}")
    print(f"  Summary txt: {output_dir / 'rerun_summary.txt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
