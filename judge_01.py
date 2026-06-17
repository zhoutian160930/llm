# -*- coding: utf-8 -*-
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw, ImageFont

# vLLM环境变量
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# ====================== 全局配置 ======================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNSUPPORTED_SEGMENTATION_WARNED = False
# 强制模型输出纯JSON的系统提示（严格约束）
DEFAULT_SYSTEM_PROMPT = """你是一名严谨的图像实例分割质量评审专家。现在给你一张原始图像及其分割标注可视化结果。图中每个分割实例都有一个唯一的 viz_index 编号，mask 通常以彩色半透明区域、轮廓或填充形式叠加在原图上。你的任务是逐一检查每个实例的分割质量，并给出严格、可解释、可复核的评分结果。

请注意：你的目标不是判断物体类别是否正确，而是判断当前 mask 是否准确、完整、独立、稳定地覆盖了对应目标。请基于图像内容、mask 轮廓、目标边界、遮挡关系和实例关系进行判断。

你必须按照以下五个维度对每个实例进行评分，总分为 100 分。

【一、评分维度与权重】

评分维度 | 权重 | 核心问题
边界贴合度 | 35分 | mask 是否贴合目标真实边界，是否存在明显边界偏移、外溢、背景误分割、边界过粗或过细。
目标完整性 | 20分 | mask 是否完整覆盖目标主体，是否漏掉关键区域，是否只覆盖了局部。
实例独立性 | 20分 | mask 是否只对应一个独立实例，是否把多个目标合并，或把同一目标错误拆分。
复杂场景适应性 | 15分 | 在遮挡、重叠、褶皱、透明、反光、低对比度、运动模糊等情况下，mask 是否仍然合理。
判断可靠性 | 10分 | 图像是否清晰，目标是否可判定，你对当前评分是否有足够把握，理由是否明确。

每个实例的最终得分为：

总分 = 边界贴合度 + 目标完整性 + 实例独立性 + 复杂场景适应性 + 判断可靠性

请给出 0–100 的整数分或一位小数分。

【二、详细评分标准】

1. 边界贴合度，满分 35 分。

该维度评估 mask 与目标真实轮廓之间的匹配程度。重点检查 mask 是否贴合目标边缘，是否越界覆盖背景，是否漏掉边缘区域，是否在边界处明显偏移。

评分标准如下：
- 33–35 分：边界高度贴合。mask 与目标轮廓基本一致，边缘平滑且准确，只有极少量像素级误差，不影响目标识别和使用。
- 30–32 分：边界较好。整体贴合目标，存在少量轻微偏移、边缘锯齿或小范围外溢，但不影响整体质量。
- 26–29 分：边界基本可用。主体轮廓正确，但局部存在明显偏移、边缘不贴合、少量背景误分割或轻微缺边。
- 20–25 分：边界问题较明显。mask 在多个局部区域偏离真实边界，存在明显外溢、边缘缺失或背景混入，但仍能看出目标主体。
- 15–19 分：边界质量较差。mask 与目标边缘大范围不一致，明显覆盖背景或漏掉目标边缘，分割结果只能勉强对应目标。
- 8–14 分：边界严重错误。mask 大量偏离真实目标，覆盖了大量背景或其他物体，仅有少部分与目标重合。
- 0–7 分：边界基本无效。mask 几乎没有贴合目标，明显标错区域，或 mask 与目标无关。

2. 目标完整性，满分 20 分。

该维度评估 mask 是否覆盖了目标应有的完整区域。重点检查目标主体、边角、细长结构、被遮挡后仍可见区域是否被合理覆盖。

评分标准如下：
- 19–20 分：目标完整覆盖。mask 覆盖目标全部可见主体，没有明显遗漏。
- 17–18 分：基本完整。仅漏掉很小的边角、细碎区域或不重要部分，不影响整体识别。
- 14–16 分：轻微缺失。目标主体基本完整，但存在局部区域漏标，例如边缘、角落、局部折叠或细长部分缺失。
- 10–13 分：明显缺失。目标仍可识别，但缺失面积较大，约占目标可见区域的 15%–30%。
- 6–9 分：严重缺失。mask 只覆盖了目标的一部分，缺失面积超过 30%，目标整体形状不完整。
- 1–5 分：主体基本缺失。mask 仅覆盖极小区域或局部碎片，无法表达目标主体。
- 0 分：未覆盖目标，或 mask 为空，或完全标到其他区域。

3. 实例独立性，满分 20 分。

该维度评估一个 mask 是否只对应一个独立目标实例。重点检查多个目标之间是否发生粘连、合并、交叉覆盖，也检查同一目标是否被错误拆成多个实例。

评分标准如下：
- 19–20 分：实例划分清楚。一个 mask 对应一个目标，没有误合并、误拆分或明显串标。
- 17–18 分：实例基本独立。存在轻微接触、边界相邻或小范围粘连，但不影响实例判断。
- 14–16 分：轻微实例混淆。mask 局部连接到相邻目标，或同一目标被轻微拆分，但主体仍然明确。
- 10–13 分：明显实例混淆。多个相邻目标部分合并，或一个目标被明显拆成多个区域，需要人工复核。
- 6–9 分：严重实例错误。多个独立目标被大范围合并，或目标与背景/其他物体关系混乱。
- 1–5 分：实例基本不可分。mask 难以对应单个目标，多个实例严重混在一起。
- 0 分：实例标注完全错误，例如一个 mask 覆盖了多个无关物体，或完全不对应任何独立目标。

4. 复杂场景适应性，满分 15 分。

该维度评估在困难视觉条件下 mask 是否仍然合理。困难条件包括但不限于：遮挡、重叠、透明包装、反光、高光、低对比度、褶皱、阴影、运动模糊、相似颜色背景、小目标、密集排列。

评分标准如下：
- 14–15 分：复杂场景下表现稳定。即使存在遮挡、反光、褶皱或重叠，mask 仍然合理。
- 12–13 分：表现较好。复杂区域有轻微误差，但整体处理合理。
- 9–11 分：基本可接受。复杂区域存在可见错误，但没有导致严重错分。
- 6–8 分：复杂区域错误明显。遮挡、反光、低对比度或重叠区域导致较多边界错误。
- 3–5 分：复杂场景处理较差。mask 明显受透明、反光、重叠或模糊影响，质量不稳定。
- 1–2 分：几乎无法处理复杂区域。mask 与目标关系严重混乱。
- 0 分：该实例在复杂区域中完全失效，无法判断其对应目标。

如果图像场景简单，没有明显遮挡、重叠、透明、反光或低对比度问题，则该维度主要根据当前简单场景下的稳定性评分。简单场景下分割合理可给 12–15 分，不应因为场景简单而自动低分。

5. 判断可靠性，满分 10 分。

该维度评估你对当前评分是否有足够把握。重点考虑图像清晰度、目标可见性、mask 可见性、viz_index 是否清楚、是否存在难以判断的遮挡或模糊。

评分标准如下：
- 9–10 分：判断非常可靠。图像清晰，目标和 mask 都可见，评分依据充分。
- 7–8 分：判断较可靠。存在轻微模糊、遮挡或局部不清楚，但不影响总体判断。
- 5–6 分：中等可靠。存在明显不确定因素，例如目标边界不清、mask 显示不完整、低对比度或遮挡较多。
- 3–4 分：判断不太可靠。图像或 mask 信息不足，需要人工复核。
- 1–2 分：判断高度不确定。几乎无法确认 mask 是否正确，只能给出保守判断。
- 0 分：无法判断。目标、mask 或编号不可见，不能进行有效评分。

【三、实例状态判定】

根据实例总分给出 quality_status：

- 90–100：excellent。分割质量很高，可直接接受。
- 85–89：good。分割质量较好，仅有轻微误差。
- 75–84：acceptable。分割基本可用，但存在一定边界、完整性或实例问题。
- 60–74：bad 或 uncertain。若错误主要来自 mask 本身，则标为 bad；若主要因为图像模糊、遮挡、反光、低对比度导致无法确定，则标为 uncertain。
- 0–59：bad。分割质量不合格。

【四、强制 bad 规则】

只要出现以下任一情况，该实例必须判定为 bad，即使部分维度看起来还可以：

1. mask 覆盖的不是目标，而是其他物体或背景。
2. mask 与目标主体重合面积很小，明显标错。
3. 目标主体缺失超过 30%。
4. mask 大面积溢出到背景，外溢面积明显影响使用。
5. 多个独立目标被严重合并为一个实例。
6. 一个完整目标被严重拆分，当前实例无法代表该目标。
7. mask 为空、极小、破碎到无法表达目标。
8. mask 覆盖区域与 viz_index 对应目标明显不一致。
9. 透明、反光或阴影区域导致 mask 完全偏离真实目标。
10. 该实例会明显影响后续自动化检测、抓取、计数或质检使用。

【五、uncertain 判定规则】

如果 mask 本身可能是正确的，但由于图像条件导致无法确认，请判定为 uncertain，而不是直接 bad。常见情况包括：

1. 目标边界被严重遮挡，真实边界不可见。
2. 图像明显模糊，无法分辨目标真实轮廓。
3. 透明或反光材料导致目标边界不可判定。
4. mask 与目标颜色高度重叠，难以看清覆盖关系。
5. viz_index 或 mask 显示不完整，无法确认对应关系。
6. 目标只有很小部分可见，无法判断完整形状。

uncertain 实例的分数通常应在 60–74 分之间。如果完全无法判断，可给 50–60 分，并说明“信息不足，需人工复核”。

【六、单图整体判定】

在完成所有实例评分后，请给出该图片的整体判定。

单图平均分计算方式：

image_score = 所有实例总分的平均值

单图状态 image_status 的判定规则如下：

- 如果所有实例均为 acceptable、good 或 excellent，且 image_score ≥ 75，则 image_status = PASS。
- 如果存在 uncertain 实例，但没有 bad 实例，则 image_status = REVIEW。
- 如果存在任意 bad 实例，则 image_status = FAIL。
- 如果 image_score < 75，则 image_status = FAIL。
- 如果实例数量很多，但只有极少数轻微 bad，请仍然标为 FAIL，并在理由中说明 bad 实例数量和风险。

【七、输出要求】

请严格输出 JSON，不要输出 Markdown，不要输出额外解释。JSON 必须包含以下字段：

{
  "image_id": "如果输入中提供了图片编号，则填写；否则填写 unknown",
  "instance_results": [
    {
      "viz_index": "实例编号",
      "boundary_fit_score": 0,
      "object_completeness_score": 0,
      "instance_independence_score": 0,
      "complex_scene_adaptability_score": 0,
      "judgment_reliability_score": 0,
      "total_score": 0,
      "quality_status": "excellent/good/acceptable/bad/uncertain",
      "main_errors": [
        "主要问题1",
        "主要问题2"
      ],
      "reason": "用一到两句话说明评分理由，必须具体指出边界、完整性、实例关系或不确定因素。"
    }
  ],
  "image_score": 0,
  "image_status": "PASS/REVIEW/FAIL",
  "bad_count": 0,
  "uncertain_count": 0,
  "pass_count": 0,
  "summary": "用两到三句话总结该图整体分割质量，指出主要风险和是否需要人工复核。"
}

【八、评分原则】

1. 请严格评分，不要因为大多数区域看起来正确就忽略明显错误。
2. 如果 mask 对自动化应用有明显风险，应降低评分。
3. 对透明包装、反光包装、边界不清、目标重叠等情况要特别谨慎。
4. 如果分割结果只是轻微边界不准，但不影响目标主体识别，可以给 acceptable 或 good。
5. 如果分割结果会影响计数、抓取、缺陷检测或质量判断，应标为 bad 或 REVIEW。
6. 不要只给总分，必须给出五个维度的分数。
7. 不要只写“较好”“一般”，理由必须具体，例如“左侧边界外溢到背景”“底部缺失一部分目标”“两个相邻包装被合并为一个实例”。
8. 所有分数必须与理由一致。若理由中出现严重缺失、严重合并、大面积外溢，则总分不能高于 74。
9. 若判定为 bad，必须在 main_errors 中明确列出导致 bad 的关键原因。
10. 若判定为 uncertain，必须说明不确定性来自图像质量、遮挡、反光、透明、低对比度还是 mask 显示不清。

现在请根据输入图像和其中的 viz_index，对所有分割实例逐一评分，并输出严格 JSON。
# """.strip()

DEFAULT_USER_PROMPT_TEMPLATE = """
Please judge the quality of the segmentation annotations in this image.

Input information:
- image_name: {image_name}
- image_width: {image_width}
- image_height: {image_height}
- num_annotations: {num_annotations}
- categories: {categories_json}
- annotations: {annotations_json}

Explanation:
- You will see two images: the first is the original image, the second is the segmentation annotation visualization.
- In the visualization, each instance is outlined with a red contour line and labeled with a text number #idx, corresponding to annotations[*].viz_index.
- Judge only based on the segmentation contour line; do not treat the text number box as the segmentation boundary.

Judging Rules and Scoring Guide:

Use a continuous scale from 0.0 to 1.0 to reflect segmentation quality:

**Score 0.9-1.0 (excellent/good)**:
   - The contour line tightly adheres to the true edges of the target.
   - Very minor, barely noticeable deviations are acceptable (within ~50 pixels).
   - The contour follows most of the target's shape details well.
   - Small amounts of background inclusion (less than 5% of target area) are tolerable if the target boundary itself is slightly ambiguous.

**Score 0.7-0.89 (acceptable)**:
   - The contour generally follows the target shape but has visible minor deviations.
   - Small "drift" visible (50-150 pixels) in some areas, but the overall shape is still recognizable.
   - May miss some small details or include minor background areas (5-15% of target area).
   - The annotation is still useful and mostly captures the target.

**Score 0.5-0.69 (borderline)**:
   - Moderate deviations are visible (150-250 pixels offset in places).
   - The contour captures the rough shape but misses significant details or includes noticeable background (15-30%).
   - Under-segmentation or over-segmentation is present but not severe.
   - The annotation partially serves its purpose but needs improvement.

**Score 0.3-0.49 (poor)**:
   - Clear and significant deviations (250-400 pixels offset).
   - The contour only roughly outlines the target area, missing major parts or including large background regions (30-50%).
   - The shape is barely recognizable from the contour alone.
   - The annotation has limited practical value.

**Score 0.0-0.29 (bad/very poor)**:
   - Severe deviations (more than 400 pixels offset).
   - The contour is on a completely different object or misses the target entirely.
   - More than 50% of the contour covers background or misses the target.
   - The contour line is broken, discontinuous, or completely fails to represent the target shape.
   - The annotation is essentially useless.

**Uncertain cases**:
   - If the target boundary in the original image is extremely ambiguous (e.g., low contrast, overexposure, underexposure, motion blur).
   - If the target is heavily occluded (more than 30%), making the true edge difficult to see.
   - If you genuinely cannot determine how well the contour fits.
   - When marking as uncertain, still provide your best estimate score and explain why it's uncertain.

Important notes:
- Relax previous strict rules: minor imperfections are acceptable and should not automatically result in "bad" status.
- "Surrounding the target roughly" with minor edge misalignments can still achieve acceptable scores (0.6-0.8).
- Only flag as "bad" (score < 0.3) when the segmentation clearly fails to represent the target.
- When in doubt between two scores, choose the higher one to avoid being overly strict.

The output must be pure JSON in the following format:
{{
  "image_name": "{image_name}",
  "overall_status": "pass|review|fail",
  "overall_score": 0.0,
  "summary": "One-sentence summary",
  "items": [
    {{
      "annotation_id": 1,
      "viz_index": 0,
      "status": "good|bad|uncertain",
      "score": 0.0,
      "reason": "Brief reason explaining the score (mention what's good and what could be improved)",
      "suggestion": "Optional correction suggestion"
    }}
  ]
}}

Scoring guidelines for items:
- score 0.9-1.0 → status: "good"
- score 0.5-0.89 → status: "good" (acceptable quality)
- score 0.3-0.49 → status: "uncertain" or "bad" depending on whether the issues are fixable
- score 0.0-0.29 → status: "bad"

Mapping for overall_status and overall_score:
- Calculate overall_score as the average of all item scores (not a fixed value).
- If all items have score >= 0.7 → overall_status = "pass"
- If any item has score < 0.3 → overall_status = "fail"
- Otherwise → overall_status = "review"

Remember: Be reasonable and practical in your judgments. The goal is to identify truly problematic annotations, not to penalize minor imperfections.
""".strip()

# ====================== vLLM 输入处理函数（复用你的本地推理代码） ======================
def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    # 处理图片/视频信息
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

# ====================== COCO 工具函数（完全保留） ======================
def load_coco(coco_path: Path) -> dict[str, Any]:
    return json.loads(coco_path.read_text(encoding="utf-8"))

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

def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score))

def output_stem(image_id: int, file_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file_name).stem).strip("._") or "image"
    return safe_name

def load_fewshot_examples(fewshot_dir: Path | None, cache_dir: Path, max_side: int) -> list[dict[str, str]]:
    if fewshot_dir is None:
        return []
    if not fewshot_dir.exists() or not fewshot_dir.is_dir():
        raise RuntimeError(f"few-shot 示例目录不存在或不是目录: {fewshot_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, str]] = []
    for idx, path in enumerate(sorted(fewshot_dir.iterdir())):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        label = path.stem.split("_")[0].strip().lower()
        if label not in {"good", "medium", "bad", "uncertain", "review"}:
            print(f"[WARN] 跳过 few-shot 示例 {path.name}: 文件名前缀需为 good_/medium_/bad_ 等标签。")
            continue
        with Image.open(path) as im:
            resized = resize_for_upload(im.convert("RGB"), max_side)
        cached_path = cache_dir / f"{idx:02d}_{label}.jpg"
        resized.save(cached_path, quality=92)
        examples.append({"label": label, "path": str(cached_path), "source_path": str(path)})

    if not examples:
        print(f"[WARN] few-shot 示例目录为空或没有可识别标签图片: {fewshot_dir}")
    return examples

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

def find_image_path(images_dir: Path, file_name: str) -> Path | None:
    candidate = images_dir / file_name
    if candidate.exists():
        return candidate
    matches = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name == Path(file_name).name]
    return matches[0] if matches else None

def _polygon_list_from_segmentation(segmentation: Any) -> list[list[float]]:
    global UNSUPPORTED_SEGMENTATION_WARNED
    if isinstance(segmentation, dict):
        if not UNSUPPORTED_SEGMENTATION_WARNED:
            print("[WARN] 检测到 RLE segmentation，当前版本不支持，相关标注将只显示编号且不会绘制轮廓。")
            UNSUPPORTED_SEGMENTATION_WARNED = True
        return []
    if not isinstance(segmentation, list):
        return []
    out: list[list[float]] = []
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        out.append([float(x) for x in poly])
    return out

def scale_segmentation(segmentation: Any, scale_x: float, scale_y: float) -> Any:
    polygons = _polygon_list_from_segmentation(segmentation)
    if not polygons:
        return segmentation
    scaled: list[list[float]] = []
    for poly in polygons:
        scaled_poly: list[float] = []
        for i in range(0, len(poly), 2):
            scaled_poly.append(poly[i] * scale_x)
            scaled_poly.append(poly[i + 1] * scale_y)
        scaled.append(scaled_poly)
    return scaled

def ann_to_mask(ann: dict[str, Any], width: int, height: int) -> Image.Image:
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for flat_poly in _polygon_list_from_segmentation(seg):
            pts = [(flat_poly[i], flat_poly[i + 1]) for i in range(0, len(flat_poly), 2)]
            draw.polygon(pts, fill=255)
        return mask
    return Image.new("L", (width, height), 0)

def color_for_index(idx: int) -> tuple[int, int, int]:
    rnd = random.Random(idx * 7919 + 17)
    return (rnd.randint(40, 235), rnd.randint(40, 235), rnd.randint(40, 235))

def build_overlay(image: Image.Image, anns: list[dict[str, Any]], render_width: int | None = None, render_height: int | None = None) -> Image.Image:
    base = image.convert("RGB")
    draw = ImageDraw.Draw(base, "RGBA")
    render_width = render_width or base.width
    render_height = render_height or base.height
    scale_x = base.width / float(render_width)
    scale_y = base.height / float(render_height)
    scale = min(scale_x, scale_y)
    font_size = max(12, int(round(20 * scale)))
    label_height = max(16, int(round(22 * scale)))
    label_width = max(72, int(round(90 * scale)))
    contour_width = max(3, int(round(5 * scale)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for ann in anns:
        idx = int(ann["viz_index"])
        scaled_segmentation = scale_segmentation(ann.get("segmentation"), scale_x, scale_y)
        for flat_poly in _polygon_list_from_segmentation(scaled_segmentation):
            pts = [(flat_poly[i], flat_poly[i + 1]) for i in range(0, len(flat_poly), 2)]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=(255, 0, 0, 255), width=contour_width)

        bbox = ann.get("bbox") or [0, 0, 0, 0]
        x, y, _, _ = [float(v) for v in bbox]
        x *= scale_x
        y *= scale_y
        label = f"#{idx}"
        tx = max(0, int(x))
        ty = max(0, int(y) - label_height)
        draw.rectangle((tx, ty, tx + label_width, ty + label_height), fill=(0, 0, 0, 200))
        draw.text((tx + 3, ty + 2), label, fill=(255, 255, 255, 255), font=font)
    return base

def make_user_prompt(
    image_name: str,
    width: int,
    height: int,
    categories: dict[int, str],
    anns: list[dict[str, Any]],
    template: str | None = None,
) -> str:
    compact_anns: list[dict[str, Any]] = []
    present_categories: set[int] = set()
    for a in anns:
        category_id = int(a.get("category_id", -1))
        present_categories.add(category_id)
        compact_anns.append({
            "annotation_id": int(a["id"]),
            "viz_index": int(a["viz_index"]),
            "category_name": categories.get(category_id, "unknown"),
        })
    category_items = [
        {"id": cid, "name": categories.get(cid, "unknown")}
        for cid in sorted(present_categories)
    ]
    tmpl = template if template is not None else DEFAULT_USER_PROMPT_TEMPLATE
    return tmpl.format(
        image_name=image_name, image_width=width, image_height=height,
        num_annotations=len(compact_anns), categories_json=json.dumps(category_items, ensure_ascii=False),
        annotations_json=json.dumps(compact_anns, ensure_ascii=False)
    )

# def normalize_judge_result(raw: dict[str, Any], anns: list[dict[str, Any]], pass_threshold: float) -> dict[str, Any]:
#     ann_by_id = {int(a["id"]): a for a in anns}
#     ann_by_viz = {int(a["viz_index"]): a for a in anns}
#     normalized_by_ann_id: dict[int, dict[str, Any]] = {}

#     for item in raw.get("items", []):
#         ann_id = item.get("annotation_id")
#         viz_index = item.get("viz_index")
#         match = None
#         if isinstance(ann_id, (int, float)) and int(ann_id) in ann_by_id:
#             match = ann_by_id[int(ann_id)]
#         elif isinstance(viz_index, (int, float)) and int(viz_index) in ann_by_viz:
#             match = ann_by_viz[int(viz_index)]
#         if match is None:
#             continue

#         score = clamp_score(item.get("score", 0.0))
#         raw_status = str(item.get("status", "")).strip().lower()
#         if raw_status not in {"good", "bad", "uncertain"}:
#             status = "good" if score >= pass_threshold else "bad"
#         else:
#             status = raw_status

#         if status == "good" and score < pass_threshold:
#             score = pass_threshold
#         elif status == "bad" and score >= pass_threshold:
#             score = max(0.0, pass_threshold - 1e-4)

#         normalized = {
#             "annotation_id": int(match["id"]),
#             "viz_index": int(match["viz_index"]),
#             "status": status,
#             "score": round(score, 4),
#             "reason": str(item.get("reason", "")).strip(),
#             "suggestion": str(item.get("suggestion", "")).strip(),
#             "category_id": int(match.get("category_id", -1)),
#         }

#         existing = normalized_by_ann_id.get(normalized["annotation_id"])
#         if existing is None:
#             normalized_by_ann_id[normalized["annotation_id"]] = normalized
#             continue

#         priority = {"bad": 2, "uncertain": 1, "good": 0}
#         existing_rank = priority[existing["status"]]
#         normalized_rank = priority[normalized["status"]]
#         if normalized_rank > existing_rank:
#             normalized_by_ann_id[normalized["annotation_id"]] = normalized
#         elif normalized_rank == existing_rank:
#             if normalized["status"] == "good" and normalized["score"] > existing["score"]:
#                 normalized_by_ann_id[normalized["annotation_id"]] = normalized
#             elif normalized["status"] in {"bad", "uncertain"} and normalized["score"] < existing["score"]:
#                 normalized_by_ann_id[normalized["annotation_id"]] = normalized

#     for ann in anns:
#         annotation_id = int(ann["id"])
#         if annotation_id not in normalized_by_ann_id:
#             normalized_by_ann_id[annotation_id] = {
#                 "annotation_id": annotation_id,
#                 "viz_index": int(ann["viz_index"]),
#                 "status": "uncertain",
#                 "score": 0.0,
#                 "reason": "model_missing_item",
#                 "suggestion": "请人工复核",
#                 "category_id": int(ann.get("category_id", -1)),
#             }

#     normalized_items = sorted(normalized_by_ann_id.values(), key=lambda x: x["viz_index"])
#     overall_score = sum(x["score"] for x in normalized_items) / len(normalized_items) if normalized_items else 0.0
#     statuses = {x["status"] for x in normalized_items}
#     if "bad" in statuses:
#         overall_status = "fail"
#     elif "uncertain" in statuses:
#         overall_status = "review"
#     elif overall_score >= pass_threshold:
#         overall_status = "pass"
#     else:
#         overall_status = "review"
#     return {
#         "overall_status": overall_status,
#         "overall_score": round(overall_score, 4),
#         "summary": str(raw.get("summary", "")).strip(),
#         "items": normalized_items,
#     }

def normalize_judge_result(raw: dict[str, Any], anns: list[dict[str, Any]], pass_threshold: float) -> dict[str, Any]:
    ann_by_id = {int(a["id"]): a for a in anns}
    ann_by_viz = {int(a["viz_index"]): a for a in anns}
    normalized_by_ann_id: dict[int, dict[str, Any]] = {}

    for item in raw.get("items", []):
        ann_id = item.get("annotation_id")
        viz_index = item.get("viz_index")
        match = None
        if isinstance(ann_id, (int, float)) and int(ann_id) in ann_by_id:
            match = ann_by_id[int(ann_id)]
        elif isinstance(viz_index, (int, float)) and int(viz_index) in ann_by_viz:
            match = ann_by_viz[int(viz_index)]
        if match is None:
            continue

        # ---- 关键修改点：使用模型给出的原始分数 ----
        original_score = item.get("score")
        
        # 如果模型给出了0-1之间的有意义分数，就信任它
        if isinstance(original_score, (int, float)) and original_score > 0.0 and original_score < 1.0:
            score = clamp_score(original_score)
        else:
            # 否则，根据状态采用默认分数，但给出一定的区分度
            raw_status = str(item.get("status", "")).strip().lower()
            if raw_status == "good":
                score = 0.85  # 良好但不完美，留出进步空间
            elif raw_status == "bad":
                score = clamp_score(item.get("score", 0.0) if item.get("score", 0.0) > 0 else 0.15)  # 可进一步细分
            elif raw_status == "uncertain":
                score = 0.5   # 不确定保持中间值
            else:
                score = clamp_score(item.get("score", 0.5))
        
        # 根据分数确定状态
        raw_status = str(item.get("status", "")).strip().lower()
        if raw_status in {"good", "bad", "uncertain"}:
            status = raw_status
        else:
            if score >= pass_threshold:
                status = "good"
            elif score <= 0.3:  # 新增加的低分阈值
                status = "bad"
            else:
                status = "uncertain"

        # 确保分数和状态的一致性
        if status == "good" and score < pass_threshold:
            score = pass_threshold
        elif status == "bad" and score >= pass_threshold:
            score = max(0.0, pass_threshold - 0.01)

        normalized = {
            "annotation_id": int(match["id"]),
            "viz_index": int(match["viz_index"]),
            "status": status,
            "score": round(score, 4),
            "reason": str(item.get("reason", "")).strip(),
            "suggestion": str(item.get("suggestion", "")).strip(),
            "category_id": int(match.get("category_id", -1)),
        }

        existing = normalized_by_ann_id.get(normalized["annotation_id"])
        if existing is None:
            normalized_by_ann_id[normalized["annotation_id"]] = normalized
            continue

        # 保留第一个遇到的结果（或者你可以换成其他冲突处理逻辑）
        # 当前逻辑：保留更差的状态
        priority = {"bad": 2, "uncertain": 1, "good": 0}
        existing_rank = priority[existing["status"]]
        normalized_rank = priority[normalized["status"]]
        if normalized_rank > existing_rank:
            normalized_by_ann_id[normalized["annotation_id"]] = normalized
        elif normalized_rank == existing_rank:
            if normalized["status"] == "good" and normalized["score"] > existing["score"]:
                normalized_by_ann_id[normalized["annotation_id"]] = normalized
            elif normalized["status"] in {"bad", "uncertain"} and normalized["score"] < existing["score"]:
                normalized_by_ann_id[normalized["annotation_id"]] = normalized

    # 补充模型未返回的标注项
    for ann in anns:
        annotation_id = int(ann["id"])
        if annotation_id not in normalized_by_ann_id:
            normalized_by_ann_id[annotation_id] = {
                "annotation_id": annotation_id,
                "viz_index": int(ann["viz_index"]),
                "status": "uncertain",
                "score": 0.0,
                "reason": "model_missing_item",
                "suggestion": "请人工复核",
                "category_id": int(ann.get("category_id", -1)),
            }

    normalized_items = sorted(normalized_by_ann_id.values(), key=lambda x: x["viz_index"])
    
    # ---- 关键修改点：使用平均分而非状态映射 ----
    overall_score = sum(x["score"] for x in normalized_items) / len(normalized_items) if normalized_items else 0.0
    
    # 但总体状态判定保持不变（因为最终状态需要明确的类别）
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
        "overall_score": round(overall_score, 4),  # 现在这会是真实的平均分
        "summary": str(raw.get("summary", "")).strip(),
        "items": normalized_items,
    }

# ====================== 核心改造：本地vLLM模型调用函数 ======================
def call_local_vllm_model(
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    system_prompt: str,
    user_prompt: str,
    original_img_path: str,
    overlay_img_path: str,
    fewshot_examples: list[dict[str, str]] | None = None,
) -> str:
    """
    替换云端API：本地vLLM推理双图+文本
    :param llm: 初始化好的vLLM LLM对象
    :param processor: 模型Processor
    :param sampling_params: 生成参数
    :param original_img_path: 原图本地路径
    :param overlay_img_path: 分割可视化图本地路径
    """
    # 构造Qwen-VL支持的消息格式（few-shot参考图 + 当前双图 + 文本）
    user_content: list[dict[str, str]] = []
    if fewshot_examples:
        user_content.append({
            "type": "text",
            "text": "先阅读以下 few-shot 参考图片。这些图片分别代表不同的标注质量等级，请学习它们的视觉特征，并在后续判断当前图片时沿用同一标准。"
        })
        for example in fewshot_examples:
            user_content.append({"type": "text", "text": f"参考样例标签: {example['label']}"})
            user_content.append({"type": "image", "image": example["path"]})

    user_content.extend([
        {"type": "text", "text": "下面开始判断当前样本。第一张是原图，第二张是分割可视化图。请参考前面的 few-shot 图片标准，但只输出当前样本的 JSON 结果。"},
        {"type": "image", "image": original_img_path},
        {"type": "image", "image": overlay_img_path},
        {"type": "text", "text": user_prompt},
    ])

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_content,
        }
    ]

    # 处理为vLLM输入格式
    inputs = prepare_inputs_for_vllm(messages, processor)
    # 执行推理
    outputs = llm.generate([inputs], sampling_params=sampling_params)
    # 返回生成结果
    return outputs[0].outputs[0].text.strip()

# ====================== 命令行参数（移除云端API参数） ======================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地vLLM模型 - COCO分割标注质量质检")
    parser.add_argument("--images-dir", type=Path, required=True, help="图片根目录")
    parser.add_argument("--coco-json", type=Path, required=True, help="COCO标注JSON路径")
    parser.add_argument("--output-dir", type=Path, default=Path("./qwen3.6_output"), help="输出目录")
    parser.add_argument("--model-path", type=str, default="/home/model/llm_model/qwen_4b", help="本地Qwen-VL模型路径")
    parser.add_argument("--system-prompt-file", type=Path, default=None, help="自定义系统提示文件")
    parser.add_argument("--user-prompt-file", type=Path, default=None, help="自定义用户提示文件")
    parser.add_argument("--max-images", type=int, default=0, help="最大处理图片数，0为无限制")
    parser.add_argument("--max-side", type=int, default=1600, help="当前待判定图片最大边长")
    parser.add_argument("--fewshot-dir", type=Path, default="/home/model/work/llm/fewshot", help="few-shot 参考图片目录，文件名前缀需为 good_/medium_/bad_ 等标签")
    parser.add_argument("--fewshot-max-side", type=int, default=1024, help="few-shot 参考图片最大边长")
    parser.add_argument("--pass-threshold", type=float, default=0.75, help="合格分数阈值")
    return parser.parse_args()

# ====================== 主函数 ======================
def main() -> None:
    args = parse_args()

    # 1. 加载COCO数据（完全保留）
    coco = load_coco(args.coco_json)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    categories_map = {int(c["id"]): str(c.get("name", f"class_{c['id']}")) for c in categories}
    ann_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        ann_by_image_id[int(ann.get("image_id", -1))].append(ann)

    # 2. 初始化本地vLLM模型（复用你的代码）
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        raise RuntimeError("未检测到可用 CUDA GPU，当前脚本依赖 vLLM + GPU 运行。")
    tensor_parallel_size = min(1, gpu_count)

    print(f"正在加载本地模型: {args.model_path}")
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
        trust_remote_code=True
    )
    # 生成参数：温度0保证输出稳定，强制JSON格式
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2048,
        top_k=-1,
        stop_token_ids=[],
    )

    # 3. 输出目录初始化
    out_json_dir = args.output_dir / "json"
    out_vis_dir = args.output_dir / "vis"
    out_input_dir = args.output_dir / "inputs"
    out_fewshot_dir = args.output_dir / "fewshot_cache"
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_vis_dir.mkdir(parents=True, exist_ok=True)
    out_input_dir.mkdir(parents=True, exist_ok=True)

    # 4. 提示词加载
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip() if args.system_prompt_file else DEFAULT_SYSTEM_PROMPT
    user_prompt_template = args.user_prompt_file.read_text(encoding="utf-8") if args.user_prompt_file else None
    fewshot_examples = load_fewshot_examples(args.fewshot_dir, out_fewshot_dir, args.fewshot_max_side)

    # 5. 批量处理图片（核心逻辑保留，仅替换模型调用）
    target_images = [img for img in images if int(img.get("id", -1)) in ann_by_image_id]
    if args.max_images > 0:
        target_images = target_images[:args.max_images]
    if not target_images:
        raise RuntimeError("COCO中未找到带标注的图片")

    report_images = []
    all_scores = []
    count_pass = count_review = count_fail = 0
    skipped_images = []

    for img in target_images:
        image_id = int(img["id"])
        file_name = str(img["file_name"])
        width, height = int(img["width"]), int(img["height"])
        img_path = find_image_path(args.images_dir, file_name)
        if not img_path:
            print(f"[WARN] 未找到图片: {file_name}")
            skipped_images.append({"image_id": image_id, "file_name": file_name, "error": "image_not_found"})
            continue

        try:
            anns = []
            for idx, ann in enumerate(sorted(ann_by_image_id[image_id], key=lambda x: int(x.get("id", 0)))):
                ann_copy = dict(ann)
                ann_copy["viz_index"] = idx
                anns.append(ann_copy)

            stem = output_stem(image_id, file_name)

            with Image.open(img_path) as im:
                original = im.convert("RGB")
            resized_original = resize_for_upload(original, args.max_side)
            input_path = out_input_dir / f"{stem}.jpg"
            resized_original.save(input_path, quality=92)
            overlay = build_overlay(resized_original, anns, render_width=width, render_height=height)
            vis_path = out_vis_dir / f"{stem}.jpg"
            overlay.save(vis_path, quality=92)

            prompt = make_user_prompt(
                image_name=file_name, width=resized_original.width, height=resized_original.height,
                categories=categories_map, anns=anns, template=user_prompt_template
            )

            print(f"正在推理图片: {file_name}")
            raw_text = call_local_vllm_model(
                llm=llm,
                processor=processor,
                sampling_params=sampling_params,
                system_prompt=system_prompt,
                user_prompt=prompt,
                original_img_path=str(input_path),
                overlay_img_path=str(vis_path),
                fewshot_examples=fewshot_examples,
            )

            parsed = extract_json_block(raw_text)
            judged = normalize_judge_result(parsed, anns=anns, pass_threshold=args.pass_threshold)

            per_image_output = {
                "image": {
                    "id": image_id,
                    "file_name": file_name,
                    "path": str(img_path),
                    "width": width,
                    "height": height,
                    "model_input_path": str(input_path),
                    "model_input_width": resized_original.width,
                    "model_input_height": resized_original.height,
                },
                "judge": judged,
                "prompt": {
                    "system": system_prompt,
                    "user": prompt,
                    "fewshot_examples": fewshot_examples,
                },
                "raw_model_output": raw_text,
            }
            out_file = out_json_dir / f"{stem}.json"
            out_file.write_text(json.dumps(per_image_output, ensure_ascii=False, indent=2), encoding="utf-8")

            overall_status = judged["overall_status"]
            overall_score = float(judged["overall_score"])
            all_scores.append(overall_score)
            if overall_status == "pass":
                count_pass += 1
            elif overall_status == "fail":
                count_fail += 1
            else:
                count_review += 1

            report_images.append({
                "image_id": image_id,
                "file_name": file_name,
                "overall_status": overall_status,
                "overall_score": overall_score,
                "num_items": len(judged["items"]),
                "json_path": str(out_file),
                "vis_path": str(vis_path),
            })
            print(f"[完成] {file_name} | 状态: {overall_status} | 分数: {overall_score:.3f}")
        except Exception as exc:
            print(f"[WARN] 跳过图片 {file_name}: {exc}")
            skipped_images.append({"image_id": image_id, "file_name": file_name, "error": str(exc)})
            continue

    # 生成总结报告
    mean_score = sum(all_scores)/len(all_scores) if all_scores else 0.0
    report = {
        "summary": {"total_images": len(report_images), "pass": count_pass, "review": count_review, "fail": count_fail,
                   "skipped": len(skipped_images), "mean_overall_score": round(mean_score,4), "model_path": args.model_path,
                   "coco_json": str(args.coco_json), "pass_threshold": args.pass_threshold},
        "images": report_images,
        "skipped_images": skipped_images,
    }
    report_path = args.output_dir / "judge_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n全部处理完成！报告路径: {report_path}")

if __name__ == "__main__":
    main()