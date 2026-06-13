#!/usr/bin/env python3
"""
最小测试: 验证 Qwen3-VL 能否描述单张图片
运行环境: conda activate vllm_new
用法:
  python /home/model/work/llm/test_qwen_describe.py
"""

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def main():
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    MODEL_PATH = "/home/model/llm_model/qwen_4b"
    TEST_IMAGE = "/root/pipelinetest/辣椒包堆叠/production_data/00000017_91625_901_1101_495_542_100.000000.jpg"

    print("=" * 60)
    print("Qwen3-VL 最小测试")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Image: {TEST_IMAGE}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    print("=" * 60)

    # Check image
    img = Image.open(TEST_IMAGE).convert("RGB")
    print(f"\n[OK] Image loaded: {img.size}")

    # Load model
    print(f"\n[1] Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print(f"[2] Loading LLM...")
    tensor_parallel_size = min(1, torch.cuda.device_count())
    llm = LLM(
        model=MODEL_PATH,
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

    print(f"[3] Testing describe...")

    # Test 1: full image describe
    print(f"\n--- Test 1: Describe full image ---")
    messages = [
        {"role": "system", "content": "你是一个物体描述专家。用一句话描述图片内容，控制在20字以内。"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": TEST_IMAGE},
                {"type": "text", "text": "请描述图片内容。"},
            ],
        },
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        print(f"  Chat template applied OK")
        print(f"  Image inputs count: {len(image_inputs) if image_inputs else 0}")

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        inputs = {"prompt": text, "multi_modal_data": mm_data}

        print(f"  Calling llm.generate()...")
        outputs = llm.generate([inputs], sampling_params=sampling_params)
        result = outputs[0].outputs[0].text.strip()
        print(f"  [OK] Result: {result}")

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: cropped region describe
    print(f"\n--- Test 2: Describe cropped bbox ---")
    x, y, w, h = 1380, 729, 758, 642
    crop = img.crop((x, y, x + w, y + h))
    crop_path = "/tmp/test_crop_qwen.jpg"
    crop.save(crop_path, quality=92)
    print(f"  Crop: ({x},{y},{w},{h}) -> {crop.size}, saved to {crop_path}")

    messages2 = [
        {"role": "system", "content": "你是一个物体描述专家。用一句话描述框内物体的外观特征（颜色、形状、材质），控制在20字以内，只输出描述。"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop_path},
                {"type": "text", "text": "请描述这个物体。"},
            ],
        },
    ]

    try:
        text2 = processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        image_inputs2, _ = process_vision_info(messages2)
        mm_data2 = {"image": image_inputs2} if image_inputs2 else {}
        inputs2 = {"prompt": text2, "multi_modal_data": mm_data2}

        print(f"  Calling llm.generate()...")
        outputs2 = llm.generate([inputs2], sampling_params=sampling_params)
        result2 = outputs2[0].outputs[0].text.strip()
        print(f"  [OK] Result: {result2}")

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("[DONE] All tests completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
