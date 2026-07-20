# -*- coding: utf-8 -*-
"""
filter_cups.py
==============
使用本地 Qwen-VL + vLLM 批量筛选「主体为透明杯子或碗」的图片。

输入: 一个图片目录(递归扫描), 不需要 COCO JSON。
输出:
  <output-dir>/kept/             - 被判定为「主体是透明杯/碗」的图片(默认复制, 加 --move 改为移动)
  <output-dir>/_resized_inputs/  - 缩放后喂给模型的中间图, 用于排查误判
  <output-dir>/manifest.json     - 每张图的判定明细 + 模型原始输出

复用 judge_01.py 的: 模型加载、prepare_inputs_for_vllm、extract_json_block、resize_for_upload。
新写的只有: 目录扫描 + batch 推理 + 复制落盘。

典型用法:
  python filter_cups.py \\
    --images-dir /path/to/images \\
    --output-dir ./cup_output

  # 大批量时调大 batch
  python filter_cups.py --images-dir /path/to/images --batch-size 16

  # 移动而非复制(原目录会清空被判定为保留的图)
  python filter_cups.py --images-dir /path/to/images --move
"""
import os
# vLLM 多进程模式, 必须在 import vllm 之前设置
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ====================== Prompts ======================
# 难点: 数据集中「透明杯/碗」和「窗户/玻璃门」常同时出现, 都是透明材质。
# Prompt 必须显式按 [形状 + 透明度 + 是否主体] 三个维度判断。
DEFAULT_SYSTEM_PROMPT = """你是一名图像内容识别助手。任务: 判断图片是否应当被保留。

【保留条件】(必须同时满足)
1. 图片的主体(占据视觉中心、面积最大或最显眼的物体)是杯子或碗
2. 该杯子/碗是透明材质(玻璃、透明塑料、亚克力等)

具体包括:
- 透明玻璃杯、透明塑料杯、高脚杯、酒杯
- 透明玻璃碗、透明塑料碗、沙拉碗、玻璃汤碗
- 即使有少量装饰(花纹、彩色底座)但整体透明也算

【不保留的情况】
- 主体是窗户、玻璃门、镜子(虽然是玻璃/反射, 但不是容器)
- 主体是不透明的杯子/碗(陶瓷、金属、木头、纸等非透明材质)
- 主体是瓶子(细长、口部收窄, 不是杯/碗形状)
- 主体是花瓶(细长、装饰性强)
- 主体是其他物品(食品、家具、人、动物等)
- 透明杯/碗 只出现在背景中, 不是主体

【边界情况】
- 图中同时有透明杯/碗和其他物品: 若透明杯/碗是主体则保留, 若只是背景小物则不保留
- 多个透明杯/碗中混杂其他物品: 只要主体(最大/最显眼)是透明杯/碗就保留
- 反光强烈看不清是否透明: confidence=low, 但若形状明确是透明杯/碗则倾向于保留

严格只输出 JSON, 不要输出 Markdown, 不要输出额外解释。"""

DEFAULT_USER_PROMPT_TEMPLATE = """
请判断这张图片是否应该保留。

判定标准: 图片的主体必须是「透明的杯子或碗」(玻璃/透明塑料材质的杯状或碗状容器)。

严格输出如下 JSON:
{{
  "keep": true,
  "main_object_type": "cup",
  "is_transparent": true,
  "confidence": "high",
  "reason": "一句话中文说明"
}}

字段要求:
- keep: boolean, true=保留(主体是透明杯/碗), false=不保留
- main_object_type: 主体的类型, 从 "cup" / "bowl" / "window" / "bottle" / "vase" / "other" 中选一个
- is_transparent: boolean, 主体是否为透明材质
- confidence: "high" / "medium" / "low"
- reason: 一句话说明, 例如"主体是一个透明玻璃碗"或"主体是窗户, 不是容器"
""".strip()

# ====================== 复用自 judge_01.py ======================
def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs,
    }


def extract_json_block(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def resize_for_upload(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    w, h = image.size
    long_side = max(w, h)
    if long_side <= max_side:
        return image
    scale = max_side / float(long_side)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return image.resize(new_size, resample=Image.Resampling.LANCZOS)


# ====================== 新逻辑: 筛选主体为透明杯/碗 ======================
def scan_images(images_dir: Path) -> list[Path]:
    """递归扫描目录下所有图片, 按路径排序保证可复现。"""
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def build_messages(system_prompt: str, user_prompt: str, image_path: str) -> list[dict[str, Any]]:
    """单图 + 文本的最简 Qwen-VL chat 结构。"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_prompt},
        ]},
    ]


def unique_dst(dst_dir: Path, src_path: Path) -> Path:
    """同名冲突时附加 8 位 hash 后缀, 避免覆盖。"""
    dst = dst_dir / src_path.name
    if not dst.exists():
        return dst
    h = hashlib.md5(str(src_path).encode("utf-8")).hexdigest()[:8]
    return dst_dir / f"{src_path.stem}_{h}{src_path.suffix}"


def _coerce_bool(value: Any) -> bool:
    """容错解析 boolean: 接受 bool / 'true'/'false'/'yes'/'1' 等。"""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="使用本地 Qwen-VL 筛选主体为透明杯/碗的图片")
    p.add_argument("--images-dir", type=Path, required=True, help="图片根目录, 递归扫描")
    p.add_argument("--output-dir", type=Path, default=Path("./cup_output"), help="输出目录")
    p.add_argument("--model-path", type=str, default="/home/model/llm_model/qwen_4b", help="本地 Qwen-VL 模型路径")
    p.add_argument("--system-prompt-file", type=Path, default=None, help="自定义 system prompt 文件")
    p.add_argument("--user-prompt-file", type=Path, default=None, help="自定义 user prompt 文件")
    p.add_argument("--max-images", type=int, default=0, help="最多处理图片数, 0=不限")
    p.add_argument("--max-side", type=int, default=1280, help="图片缩放最大边长, 降低可加快推理")
    p.add_argument("--batch-size", type=int, default=8, help="批量推理大小")
    p.add_argument("--move", action="store_true", help="移动而非复制(默认复制, 可逆)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. 加载模型 (与 judge_01.py 一致, max_model_len 和 max_tokens 调小加速)
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        raise RuntimeError("未检测到可用 CUDA GPU, 本脚本依赖 vLLM + GPU。")
    print(f"正在加载本地模型: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,
        tensor_parallel_size=min(1, gpu_count),
        seed=0,
        max_model_len=8192,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        disable_log_stats=True,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,  # yes/no 类任务不需要长输出
        top_k=-1,
        stop_token_ids=[],
    )

    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8").strip()
        if args.system_prompt_file else DEFAULT_SYSTEM_PROMPT
    )
    user_prompt = (
        args.user_prompt_file.read_text(encoding="utf-8").strip()
        if args.user_prompt_file else DEFAULT_USER_PROMPT_TEMPLATE
    )

    # 2. 输出目录
    kept_dir = args.output_dir / "kept"
    cache_dir = args.output_dir / "_resized_inputs"
    kept_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 3. 扫描图片
    all_images = scan_images(args.images_dir)
    if args.max_images > 0:
        all_images = all_images[:args.max_images]
    if not all_images:
        raise RuntimeError(f"在 {args.images_dir} 中没有找到图片")
    print(f"共发现 {len(all_images)} 张图片待处理, batch_size={args.batch_size}")

    # 4. 批量推理 + 落盘
    manifest: list[dict[str, Any]] = []
    count_keep = 0
    count_skip = 0
    count_error = 0

    for batch_start in range(0, len(all_images), args.batch_size):
        batch_paths = all_images[batch_start:batch_start + args.batch_size]

        # 4.1 预处理: resize + 缓存, 构造 vLLM 输入
        vllm_inputs: list[tuple[Path, Path, dict]] = []
        for idx, img_path in enumerate(batch_paths):
            global_idx = batch_start + idx
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", img_path.stem).strip("._") or "image"
            cached_path = cache_dir / f"{global_idx:06d}_{stem}.jpg"
            try:
                with Image.open(img_path) as im:
                    resized = resize_for_upload(im.convert("RGB"), args.max_side)
                resized.save(cached_path, quality=92)
            except Exception as exc:
                print(f"[WARN] 无法读取 {img_path}: {exc}")
                manifest.append({
                    "image_path": str(img_path),
                    "keep": None,
                    "error": f"image_read_failed: {exc}",
                })
                count_error += 1
                continue

            messages = build_messages(system_prompt, user_prompt, str(cached_path))
            try:
                inputs = prepare_inputs_for_vllm(messages, processor)
            except Exception as exc:
                print(f"[WARN] 构造输入失败 {img_path}: {exc}")
                manifest.append({
                    "image_path": str(img_path),
                    "keep": None,
                    "error": f"prepare_input_failed: {exc}",
                })
                count_error += 1
                continue
            vllm_inputs.append((img_path, cached_path, inputs))

        if not vllm_inputs:
            continue

        # 4.2 批量推理: 一次 generate 多张图, 显著提升吞吐
        try:
            outputs = llm.generate([it[2] for it in vllm_inputs], sampling_params=sampling_params)
        except Exception as exc:
            print(f"[WARN] 批量推理失败 (start={batch_start}): {exc}")
            for img_path, _cached_path, _ in vllm_inputs:
                manifest.append({
                    "image_path": str(img_path),
                    "keep": None,
                    "error": f"generate_failed: {exc}",
                })
                count_error += 1
            continue

        # 4.3 解析每张图的结果 + 落盘
        for (img_path, cached_path, _), out in zip(vllm_inputs, outputs):
            raw_text = out.outputs[0].text.strip()
            try:
                parsed = extract_json_block(raw_text)
                keep = _coerce_bool(parsed.get("keep"))
                main_object_type = str(parsed.get("main_object_type", "unknown")).strip().lower()
                is_transparent = _coerce_bool(parsed.get("is_transparent"))
                confidence = str(parsed.get("confidence", "unknown")).strip().lower()
                reason = str(parsed.get("reason", "")).strip()
            except Exception as exc:
                print(f"[WARN] JSON 解析失败 {img_path.name}: {exc}")
                manifest.append({
                    "image_path": str(img_path),
                    "keep": None,
                    "error": f"parse_failed: {exc}",
                    "raw_model_output": raw_text,
                })
                count_error += 1
                continue

            record = {
                "image_path": str(img_path),
                "keep": keep,
                "main_object_type": main_object_type,
                "is_transparent": is_transparent,
                "confidence": confidence,
                "reason": reason,
                "model_input_path": str(cached_path),
                "raw_model_output": raw_text,
            }
            manifest.append(record)

            # 落盘: 复制/移动需要保留的图
            if keep:
                dst = unique_dst(kept_dir, img_path)
                if args.move:
                    shutil.move(str(img_path), str(dst))
                else:
                    shutil.copy2(str(img_path), str(dst))
                count_keep += 1
                tag = "KEEP"
            else:
                count_skip += 1
                tag = "    "
            print(f"[{tag}] {img_path.name} | type={main_object_type} | transparent={is_transparent} | conf={confidence} | {reason[:50]}")

    # 5. 写 manifest
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "summary": {
                "total": len(all_images),
                "kept": count_keep,
                "skipped": count_skip,
                "error": count_error,
                "model_path": args.model_path,
                "images_dir": str(args.images_dir),
                "kept_dir": str(kept_dir),
            },
            "items": manifest,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n完成! 共 {len(all_images)} 张 | 保留 {count_keep} | 跳过 {count_skip} | 错误 {count_error}")
    print(f"保留图片目录: {kept_dir}")
    print(f"详细清单: {manifest_path}")


if __name__ == "__main__":
    main()
