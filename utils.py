"""
Batch Debug Output Plugin - Utility Functions
ComfyUI 批量调试出图插件 - 工具函数
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import itertools
import math
import re
import os


def tensor_to_pil(image_tensor):
    """
    Convert a ComfyUI image tensor to PIL Image.
    Handles [H,W,C], [1,H,W,C], and [B,H,W,C] (takes first of batch).

    ComfyUI 图像张量转 PIL 图像
    """
    if image_tensor.dim() == 4:
        if image_tensor.shape[0] > 1:
            image_tensor = image_tensor[0]
        else:
            image_tensor = image_tensor.squeeze(0)
    arr = 255.0 * image_tensor.cpu().numpy()
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(pil_image):
    """
    Convert PIL Image to ComfyUI tensor [1, H, W, C].

    PIL 图像转 ComfyUI 张量
    """
    arr = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def cartesian_product_sweep(config, skip_prompts=False):
    """
    Build the list of sweep parameter combinations from config dict.

    Args:
        config: dict with keys:
            prompts (list[str]), loras (list[{name, weights, clip_mult}]),
            cfgs (list[float]), seed (int), steps_list (list[int])
        skip_prompts: If True, ignore prompts dimension (use empty string).

    Returns:
        list of dicts, each with: prompt, lora_weights (list of floats), cfg, seed, steps
    """
    prompts = config.get("prompts", []) if not skip_prompts else [""]
    loras = config.get("loras", [{"name": "", "weights": [0.0], "clip_mult": 1.0}])
    cfgs = config.get("cfgs", [7.0])
    seed = config.get("seed", 42)
    steps_list = config.get("steps_list", [20])

    # Build LoRA weight combinations (cartesian product across multiple LoRAs)
    lora_weight_lists = [l["weights"] for l in loras if l.get("name")]
    if not lora_weight_lists:
        lora_weight_lists = [[0.0]]

    combinations = []
    for prompt in prompts:
        for lora_combo in itertools.product(*lora_weight_lists):
            for cfg_val in cfgs:
                for steps in steps_list:
                    if "<var>" in prompt:
                        prompt_resolved = prompt.replace("<var>", f"cfg{cfg_val:.1f}")
                    else:
                        prompt_resolved = prompt

                    combinations.append({
                        "prompt": prompt_resolved,
                        "lora_weights": list(lora_combo),
                        "cfg": round(cfg_val, 2),
                        "seed": int(seed),
                        "steps": int(steps),
                    })

    return combinations


def sanitize_filename(name, max_len=100):
    """
    Remove characters unsafe for filenames across OS.

    移除文件名中的非法字符
    """
    # Replace characters invalid on Windows/Linux
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', name)
    name = name.strip().replace(' ', '_')
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    return name[:max_len]


def format_label(template, record):
    """
    Safely format a label string from a metadata record.
    Falls back to key=value pairs if template has unexpected keys.

    安全地格式化标注文本，如果模板包含未知键则回退到 key=value 格式
    """
    # Truncate prompt for labels
    truncated = dict(record)
    if "prompt" in truncated and len(str(truncated["prompt"])) > 40:
        truncated["prompt"] = str(truncated["prompt"])[:37] + "..."

    try:
        return template.format(**truncated)
    except (KeyError, ValueError):
        # Fallback: show key parameters concisely
        parts = []
        for k in ["lora_info", "cfg", "steps"]:
            if k in truncated:
                parts.append(f"{k}={truncated[k]}")
        return ", ".join(parts)


def _get_font(font_size):
    """Try to load a usable font; fall back to default."""
    # Try common Windows fonts first, then Linux/Mac fonts
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttf",       # Microsoft YaHei (Chinese support)
        "C:/Windows/Fonts/simsun.ttc",     # SimSun (Chinese support)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, font_size)
        except (IOError, OSError):
            continue
    # Ultimate fallback
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def build_labeled_grid(image_tensors, labels, columns=4,
                       cell_width=256, max_grid_dimension=4096,
                       font_size=12, bg_color=(30, 30, 30),
                       label_bg_color=(50, 50, 50),
                       text_color=(255, 255, 255)):
    """
    Build a labeled grid image from a list of image tensors.

    Args:
        image_tensors: list of torch.Tensor [H,W,C] or [1,H,W,C], or single [B,H,W,C]
        labels: list of str, one per image
        columns: number of columns in the grid
        cell_width: target width of each cell in pixels
        max_grid_dimension: downscale if grid exceeds this
        font_size: font size for cell labels
        bg_color: RGB tuple for grid background
        label_bg_color: RGB tuple for label bar background
        text_color: RGB tuple for label text

    Returns:
        PIL.Image of the labeled grid

    构建带标注的网格对比图
    """
    n = len(image_tensors)
    if n == 0:
        # Return a placeholder image
        return Image.new("RGB", (256, 256), color=bg_color)

    rows = math.ceil(n / columns)

    # Step 1: Convert all tensors to PIL and resize to cell_width
    pil_images = []
    for t in image_tensors:
        pil = tensor_to_pil(t)
        # Maintain aspect ratio, compute cell_height
        aspect = pil.height / max(pil.width, 1)
        ch = max(int(cell_width * aspect), 1)
        pil = pil.resize((cell_width, ch), Image.LANCZOS)
        pil_images.append(pil)

    # Step 2: Use uniform cell_height (max across all images) for consistent grid
    cell_height = max(img.height for img in pil_images)

    # Step 3: Check grid dimensions against max, downscale if needed
    grid_w = columns * cell_width
    grid_h = rows * cell_height
    scale_factor = min(max_grid_dimension / max(grid_w, 1),
                       max_grid_dimension / max(grid_h, 1),
                       1.0)
    if scale_factor < 1.0:
        cell_width = max(int(cell_width * scale_factor), 1)
        cell_height = max(int(cell_height * scale_factor), 1)
        pil_images = [img.resize((cell_width, cell_height), Image.LANCZOS)
                      for img in pil_images]

    # Step 4: Add label bar height
    label_bar_height = font_size + 6
    canvas_w = columns * cell_width
    canvas_h = rows * (cell_height + label_bar_height)

    # Step 5: Create grid canvas
    grid = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
    font = _get_font(font_size)
    draw = ImageDraw.Draw(grid)

    # Step 6: Place images and labels
    for idx, (pil_img, label) in enumerate(zip(pil_images, labels)):
        row = idx // columns
        col = idx % columns
        x = col * cell_width
        y_base = row * (cell_height + label_bar_height)

        # Draw label background bar
        draw.rectangle(
            [x, y_base, x + cell_width, y_base + label_bar_height],
            fill=label_bg_color
        )

        # Draw label text
        actual_font = font
        if actual_font:
            draw.text((x + 2, y_base + 2), label, fill=text_color, font=actual_font)

        # Center image in its cell (handle narrower images)
        img_x = x + (cell_width - pil_img.width) // 2
        img_y = y_base + label_bar_height + (cell_height - pil_img.height) // 2
        grid.paste(pil_img, (img_x, img_y))

    return grid


def write_metadata_csv(csv_path, metadata_records):
    """
    Export sweep metadata records as CSV file.
    Suitable for downstream scoring/evaluation workflows.

    导出参数扫描记录为 CSV 文件，方便后续打分评估
    """
    import csv

    if not metadata_records:
        return

    fieldnames = [
        "index", "prompt", "lora_info", "lora_weights",
        "cfg", "seed", "steps", "sampler", "scheduler", "filename"
    ]

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(metadata_records)
    except Exception as e:
        print(f"[BatchDebug] Warning: Failed to write CSV: {e}")


def get_sampler_scheduler_options():
    """
    Try to import ComfyUI's sampler/scheduler lists.
    Falls back to sensible defaults if imports fail (e.g., standalone testing).
    """
    try:
        import comfy.samplers
        samplers = list(comfy.samplers.KSampler.SAMPLERS)
        schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
        return samplers, schedulers
    except ImportError:
        default_samplers = ["euler", "euler_ancestral", "heun", "heunpp2",
                            "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast",
                            "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
                            "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde",
                            "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
                            "ddpm", "lcm", "ipndm", "ipndm_v", "deis",
                            "res_multistep", "res_multistep_cfg",
                            "res_multistep_ancestral", "res_multistep_ancestral_cfg",
                            "gradient_estimation"]
        default_schedulers = ["normal", "karras", "exponential", "sgm_uniform",
                              "simple", "ddim_uniform", "beta", "linear_quadratic"]
        return default_samplers, default_schedulers
