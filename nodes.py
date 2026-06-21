"""
Batch Debug Output Plugin - Node Implementations
ComfyUI 批量调试出图插件 - 节点实现

Three nodes:
  1. BatchDebugConfig  - 批量调试配置 (parameter sweep setup)
  2. BatchDebugExecute - 批量调试执行 (core generation engine)
  3. BatchDebugGridSave - 批量调试保存 (grid + individual + CSV output)
"""

import torch
import json
import os
import math
import time
import sys
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# ComfyUI internal imports
import folder_paths
import comfy.sd
import comfy.utils
import comfy.samplers
import comfy.model_management
from nodes import common_ksampler

# Local utilities
from .utils import (
    tensor_to_pil,
    pil_to_tensor,
    cartesian_product_sweep,
    sanitize_filename,
    format_label,
    build_labeled_grid,
    write_metadata_csv,
    get_sampler_scheduler_options,
)


# ---------------------------------------------------------------------------
# Node 1: BatchDebugConfig - 批量调试配置
# ---------------------------------------------------------------------------
class BatchDebugConfig:
    """
    Configuration node for batch debug sweep.
    6 LoRA slots — pick from dropdown, toggle on/off, set weight sweep range per slot.
    Seed is FIXED (not scanned) for fair comparison.
    """

    N_LORA_SLOTS = 6

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["none"] + folder_paths.get_filename_list("loras")

        required = {
            # --- Prompt ---
            "prompts": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "One prompt per line. Leave empty for D站 conditioning mode."
            }),

            # --- Global LoRA CLIP multiplier ---
            "lora_weight_clip": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                "tooltip": "Global CLIP multiplier — applied to all LoRA clip weights."
            }),

            # --- CFG sweep ---
            "cfg_min": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.5}),
            "cfg_max": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 100.0, "step": 0.5}),
            "cfg_steps": ("INT", {"default": 4, "min": 1, "max": 100}),

            # --- Fixed seed ---
            "seed": ("INT", {
                "default": 42, "min": 0, "max": 0xffffffffffffffff,
                "tooltip": "Fixed seed for ALL combinations."
            }),

            # --- Steps sweep ---
            "steps_min": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "steps_max": ("INT", {"default": 30, "min": 1, "max": 10000}),
            "steps_count": ("INT", {"default": 2, "min": 1, "max": 100}),
        }

        # Add 6 LoRA slots
        for i in range(1, cls.N_LORA_SLOTS + 1):
            required[f"lora_{i}_enabled"] = ("BOOLEAN", {
                "default": False,
                "label_on": f"LoRA #{i} ON",
                "label_off": f"LoRA #{i} OFF",
            })
            required[f"lora_{i}_name"] = (lora_list, {
                "default": "none",
                "tooltip": f"Select LoRA #{i} to sweep."
            })
            required[f"lora_{i}_weight_min"] = ("FLOAT", {
                "default": 0.3, "min": -10.0, "max": 10.0, "step": 0.05,
                "tooltip": f"LoRA #{i} minimum weight."
            })
            required[f"lora_{i}_weight_max"] = ("FLOAT", {
                "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                "tooltip": f"LoRA #{i} maximum weight."
            })
            required[f"lora_{i}_weight_steps"] = ("INT", {
                "default": 3, "min": 1, "max": 20,
                "tooltip": f"LoRA #{i} number of weight steps."
            })

        return {"required": required}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("config_json",)
    FUNCTION = "build_config"
    CATEGORY = "batch_debug"
    DESCRIPTION = "6-slot LoRA sweeper — pick from dropdown, toggle ON/OFF, set weight range. 6槽LoRA扫描配置。"

    def build_config(self, **kwargs):
        # Parse prompts
        prompts_raw = kwargs.get("prompts", "")
        prompt_list = [p.strip() for p in prompts_raw.split('\n') if p.strip() and not p.strip().startswith('#')]
        if not prompt_list:
            prompt_list = [""]

        # Parse 6 LoRA slots
        loras = []
        for i in range(1, self.N_LORA_SLOTS + 1):
            enabled = kwargs.get(f"lora_{i}_enabled", False)
            name = kwargs.get(f"lora_{i}_name", "none")
            if not enabled or name == "none":
                continue
            w_min = kwargs.get(f"lora_{i}_weight_min", 0.3)
            w_max = kwargs.get(f"lora_{i}_weight_max", 1.0)
            w_steps = kwargs.get(f"lora_{i}_weight_steps", 3)
            weights = np.linspace(w_min, w_max, max(w_steps, 2)).round(4).tolist()
            loras.append({
                "name": name,
                "weights": weights,
                "clip_mult": 1.0,
            })
            print(f"[BatchDebug] LoRA #{i} '{name}': {w_min}→{w_max} ×{w_steps} = {len(weights)} values")

        if not loras:
            loras = [{"name": "", "weights": [0.0], "clip_mult": 1.0}]

        # Build sweeps
        lora_weight_clip = kwargs.get("lora_weight_clip", 1.0)
        cfgs = np.linspace(kwargs.get("cfg_min", 3), kwargs.get("cfg_max", 9),
                          max(kwargs.get("cfg_steps", 4), 2)).round(2).tolist()
        seed = int(kwargs.get("seed", 42))
        steps_list = [int(s) for s in np.linspace(
            kwargs.get("steps_min", 20), kwargs.get("steps_max", 30),
            max(kwargs.get("steps_count", 2), 2)
        ).round().astype(int)]

        config = {
            "prompts": prompt_list,
            "loras": loras,
            "lora_weight_clip": round(lora_weight_clip, 4),
            "cfgs": cfgs,
            "seed": seed,
            "steps_list": steps_list,
        }

        # Total count
        lora_combos = 1
        for lc in loras:
            if lc["name"]:
                lora_combos *= len(lc["weights"])
        total = len(prompt_list) * lora_combos * len(cfgs) * len(steps_list)
        print(f"[BatchDebug] {len(prompt_list)} prompts × {lora_combos} LoRA combos "
              f"× {len(cfgs)} CFGs × {len(steps_list)} steps = {total} total (seed={seed})")
        if total > 200:
            print(f"[BatchDebug] ⚠ {total} combinations — may use significant memory.")

        return (json.dumps(config, ensure_ascii=False),)


# ---------------------------------------------------------------------------
# Node 2: BatchDebugExecute - 批量调试执行
# ---------------------------------------------------------------------------
class BatchDebugExecute:
    """
    Core execution node for batch debug sweeps.
    Iterates all parameter combinations, runs sampling + VAE decode, returns batched images.

    批量调试执行节点 - 遍历所有参数组合，执行采样和 VAE 解码
    """

    def __init__(self):
        self.lora_cache = {}  # {lora_name: (lora_path, lora_sd)}

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            samplers = list(comfy.samplers.KSampler.SAMPLERS)
            schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
        except Exception:
            samplers, schedulers = get_sampler_scheduler_options()
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model for prompt encoding."}),
                "vae": ("VAE", {"tooltip": "The VAE for decoding latents to images."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning."}),
                "latent_image": ("LATENT", {"tooltip": "Input latent image (empty latent)."}),
                "config_json": ("STRING", {
                    "multiline": False,
                    "default": "{}",
                    "tooltip": "Connect from BatchDebugConfig node.",
                    "forceInput": True,
                }),
                "prompt_source": (["reencode", "conditioning"], {
                    "default": "conditioning",
                    "tooltip": "'conditioning' uses upstream CLIPTextEncode (supports AnimaPromptConverter). 'reencode' re-encodes prompts from config."
                }),
                "sampler_name": (samplers, {"default": "euler"}),
                "scheduler": (schedulers, {"default": "normal"}),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoising strength."
                }),
                "save_preview": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Preview ON",
                    "label_off": "Preview OFF",
                    "tooltip": "Save each image immediately to output/preview/ so you can watch progress in real-time."
                }),
                "preview_prefix": ("STRING", {
                    "default": "batch_debug/preview",
                    "multiline": False,
                    "tooltip": "Subfolder under ComfyUI/output/ for real-time preview images."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "metadata_json")
    OUTPUT_TOOLTIPS = (
        "Batch of all generated images [N, H, W, C].",
        "JSON metadata array with parameter records for each image."
    )
    FUNCTION = "execute_batch"
    CATEGORY = "batch_debug"
    DESCRIPTION = "Execute a batch debug sweep across parameter combinations. 执行批量调试参数扫描。"
    SEARCH_ALIASES = ["batch generate", "parameter sweep", "grid test", "批量生成", "参数测试"]

    def execute_batch(self, model, clip, vae, positive, negative, latent_image,
                      config_json, prompt_source, sampler_name, scheduler, denoise=1.0,
                      save_preview=True, preview_prefix="batch_debug/preview",
                      prompt=None, extra_pnginfo=None):
        # --- 1. Parse config ---
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError:
            raise ValueError("[BatchDebug] Invalid config_json. "
                             "Connect a BatchDebugConfig node.")

        if not config:
            raise ValueError("[BatchDebug] Config is empty.")

        use_conditioning = (prompt_source == "conditioning")
        combinations = cartesian_product_sweep(config, skip_prompts=use_conditioning)
        total = len(combinations)
        print(f"[BatchDebug] Starting sweep: {total} combinations "
              f"(mode: {prompt_source})")
        print(f"[BatchDebug] Sampler: {sampler_name}, Scheduler: {scheduler}, "
              f"Denoise: {denoise}")

        # --- 2. Pre-load all LoRAs ---
        lora_configs = config.get("loras", [{"name": "", "weights": [0.0], "clip_mult": 1.0}])
        lora_weight_clip = config.get("lora_weight_clip", 1.0)
        lora_sds = {}  # {name: state_dict}

        for lc in lora_configs:
            name = lc.get("name", "")
            if not name:
                continue
            lora_path = folder_paths.get_full_path_or_raise("loras", name)
            if name in self.lora_cache and self.lora_cache[name][0] == lora_path:
                lora_sds[name] = self.lora_cache[name][1]
            else:
                print(f"[BatchDebug] Loading LoRA: {name}")
                lora_sds[name] = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.lora_cache[name] = (lora_path, lora_sds[name])

        # --- 3. Main loop ---
        image_tensors = []
        metadata_records = []
        start_time = time.time()

        try:
            for idx, combo in enumerate(combinations):
                comfy.model_management.throw_exception_if_processing_interrupted()

                prompt_text = combo.get("prompt", "")
                lora_weights = combo["lora_weights"]  # list of floats, one per LoRA
                cfg_val = combo["cfg"]
                seed = combo["seed"]
                steps = combo["steps"]

                # --- 3a. Chain-apply all LoRAs ---
                model_used = model
                clip_used = clip
                lora_idx = 0
                for lc in lora_configs:
                    name = lc.get("name", "")
                    if not name or name not in lora_sds:
                        continue
                    weight = lora_weights[lora_idx] if lora_idx < len(lora_weights) else 1.0
                    clip_mult = lc.get("clip_mult", 1.0)
                    model_used, clip_used = comfy.sd.load_lora_for_models(
                        model_used, clip_used, lora_sds[name],
                        weight, weight * lora_weight_clip * clip_mult
                    )
                    lora_idx += 1

                # Build LoRA info string for logging
                lora_info_parts = []
                for i, lc in enumerate(lora_configs):
                    if lc.get("name") and i < len(lora_weights):
                        lora_info_parts.append(f"{lc['name'].split('.')[0]}:{lora_weights[i]:.2f}")
                lora_info = " + ".join(lora_info_parts) if lora_info_parts else "none"

                # --- 3b. Encode or reuse conditioning ---
                if use_conditioning:
                    pos_cond = positive
                    neg_cond = negative
                else:
                    if lora_sds:
                        tokens = clip_used.tokenize(prompt_text)
                        pos_cond = clip_used.encode_from_tokens_scheduled(tokens)
                    else:
                        tokens = clip.tokenize(prompt_text)
                        pos_cond = clip.encode_from_tokens_scheduled(tokens)
                    neg_cond = negative

                # --- 3c. Prepare latent ---
                latent_copy = latent_image.copy()
                latent_copy["samples"] = latent_image["samples"].clone()

                # --- 3d. Run sampler ---
                try:
                    latent_out, = common_ksampler(
                        model_used, seed, steps, cfg_val,
                        sampler_name, scheduler,
                        pos_cond, neg_cond, latent_copy,
                        denoise=denoise
                    )
                except Exception as e:
                    print(f"[BatchDebug] Error at {idx + 1}/{total}: "
                          f"cfg={cfg_val:.1f} lora=[{lora_info}] steps={steps}")
                    print(f"  Error: {e}")
                    continue

                # --- 3e. Decode ---
                latent_samples = latent_out["samples"]
                if latent_samples.is_nested:
                    latent_samples = latent_samples.unbind()[0]
                pixels = vae.decode(latent_samples)
                if len(pixels.shape) == 5:
                    pixels = pixels.reshape(-1, pixels.shape[-3],
                                           pixels.shape[-2], pixels.shape[-1])

                # --- 3e2. Real-time preview save (before collection) ---
                if save_preview:
                    try:
                        preview_dir = os.path.join(folder_paths.get_output_directory(), preview_prefix)
                        os.makedirs(preview_dir, exist_ok=True)
                        preview_tensor = pixels[0] if pixels.dim() == 4 else pixels
                        preview_arr = (255.0 * preview_tensor.cpu().numpy()).clip(0, 255).astype("uint8")
                        preview_img = Image.fromarray(preview_arr)
                        preview_path = os.path.join(preview_dir, f"preview_{idx + 1:04d}.png")
                        preview_img.save(preview_path)
                    except Exception:
                        pass  # Preview is best-effort; don't crash the sweep

                # --- 3f. Collect ---
                image_tensors.append(pixels.cpu())
                metadata_records.append({
                    "index": len(image_tensors) - 1,
                    "prompt": prompt_text,
                    "lora_info": lora_info,
                    "lora_weights": lora_weights,
                    "cfg": round(cfg_val, 2),
                    "seed": int(seed),
                    "steps": int(steps),
                    "sampler": sampler_name,
                    "scheduler": scheduler,
                })

                # --- 3g. Progress ---
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = avg_time * (total - idx - 1)
                print(f"[BatchDebug] {idx + 1}/{total} "
                      f"({elapsed:.0f}s, ~{remaining:.0f}s left): "
                      f"cfg={cfg_val:.1f} lora=[{lora_info}] steps={steps}")

        finally:
            if lora_sds:
                lora_sds.clear()
                comfy.model_management.soft_empty_cache()

        # --- 4. Final assembly ---
        total_time = time.time() - start_time
        successful = len(image_tensors)
        print(f"[BatchDebug] Sweep complete: {successful}/{total} successful "
              f"in {total_time:.0f}s ({total_time / max(successful, 1):.1f}s per image)")

        if successful == 0:
            raise RuntimeError("[BatchDebug] No images were generated successfully. "
                               "Check your parameters and model compatibility.")

        # Stack all images: [N, H, W, C]
        # Images may have different sizes, so pad to max dimensions
        max_h = max(t.shape[0] for t in image_tensors if t.dim() >= 3)
        max_w = max(t.shape[1] for t in image_tensors if t.dim() >= 3)

        padded_tensors = []
        for t in image_tensors:
            if t.dim() == 3:
                h, w, c = t.shape
                if h != max_h or w != max_w:
                    # Pad with zeros to max size
                    padded = torch.zeros(max_h, max_w, c, dtype=t.dtype)
                    padded[:h, :w, :] = t
                    padded_tensors.append(padded)
                else:
                    padded_tensors.append(t)
            elif t.dim() == 4:
                # Already has batch dim
                for j in range(t.shape[0]):
                    padded_tensors.append(t[j])
            else:
                padded_tensors.append(t)

        all_images = torch.stack(padded_tensors, dim=0)  # [N, H, W, C]

        # Serialize metadata
        metadata_json = json.dumps(metadata_records, ensure_ascii=False)

        return (all_images, metadata_json)


# ---------------------------------------------------------------------------
# Node 3: BatchDebugGridSave - 批量调试网格保存
# ---------------------------------------------------------------------------
class BatchDebugGridSave:
    """
    Output node: saves each image individually with parameter filenames,
    creates a labeled comparison grid, and exports a CSV metadata file.

    批量调试网格保存节点 - 保存单张图(带参数文件名) + 标注网格图 + CSV 元数据
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Image batch from BatchDebugExecute."
                }),
                "metadata_json": ("STRING", {
                    "multiline": False,
                    "default": "[]",
                    "tooltip": "Metadata JSON from BatchDebugExecute.",
                    "forceInput": True,
                }),
                "columns": ("INT", {
                    "default": 4, "min": 1, "max": 32,
                    "tooltip": "Number of columns in the grid."
                }),
                "cell_width": ("INT", {
                    "default": 256, "min": 64, "max": 1024,
                    "tooltip": "Width in pixels for each grid cell."
                }),
                "max_grid_dimension": ("INT", {
                    "default": 4096, "min": 512, "max": 16384,
                    "tooltip": "Maximum grid width/height; downscales if exceeded."
                }),
                "label_font_size": ("INT", {
                    "default": 16, "min": 6, "max": 48,
                    "tooltip": "Font size for cell labels in the grid."
                }),
                "label_format": ("STRING", {
                    "default": "C:{cfg:.1f} L:{lora_info}",
                    "multiline": False,
                    "tooltip": "Format string for grid labels. Keys: prompt, lora_info, cfg, seed, steps, sampler, scheduler, index."
                }),
                "filename_prefix": ("STRING", {
                    "default": "batch_debug/%date:yyyy-MM-dd%_%time:HH-mm-ss%",
                    "tooltip": "Prefix for saved files. Supports %date:format% and %time:format% placeholders."
                }),
                "save_individual": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Yes",
                    "label_off": "No",
                    "tooltip": "Save each image as a separate PNG file with parameter names."
                }),
                "save_grid": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Yes",
                    "label_off": "No",
                    "tooltip": "Save the labeled comparison grid image."
                }),
                "save_metadata_csv": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Yes",
                    "label_off": "No",
                    "tooltip": "Export sweep metadata as a CSV file for scoring."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_batch_debug"
    OUTPUT_NODE = True
    CATEGORY = "batch_debug"
    DESCRIPTION = "Save batch debug results: individual images + labeled grid + CSV metadata. 保存批量调试结果：单张图+标注网格+CSV。"
    SEARCH_ALIASES = ["debug save", "grid save", "batch save", "批量保存", "调试保存"]

    def save_batch_debug(self, images, metadata_json, columns, cell_width,
                         max_grid_dimension, label_font_size, label_format,
                         filename_prefix, save_individual, save_grid,
                         save_metadata_csv, prompt=None, extra_pnginfo=None):
        # --- Parse metadata ---
        try:
            metadata_records = json.loads(metadata_json)
        except json.JSONDecodeError:
            print("[BatchDebug] Warning: Invalid metadata_json, using empty records.")
            metadata_records = []

        n = images.shape[0]
        if metadata_records and len(metadata_records) != n:
            print(f"[BatchDebug] Warning: {n} images but {len(metadata_records)} "
                  f"metadata records. Using available data.")
            # Pad or trim metadata to match
            while len(metadata_records) < n:
                metadata_records.append({"index": len(metadata_records)})

        # --- Get save path ---
        full_output_folder, filename_base, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir,
                images[0].shape[1], images[0].shape[0]
            )

        results = []

        # --- 1. Save individual images ---
        if save_individual:
            for i in range(n):
                img_tensor = images[i]  # [H, W, C]
                pil_img = tensor_to_pil(img_tensor)

                # Build parameter-rich filename
                meta = metadata_records[i] if i < len(metadata_records) else {}
                prompt_short = sanitize_filename(
                    str(meta.get("prompt", "unknown"))[:20], max_len=20
                )
                lora_info = meta.get("lora_info", "none")
                lora_short = sanitize_filename(lora_info, max_len=40)
                cfg_v = meta.get("cfg", 0.0)
                steps_v = meta.get("steps", 0)

                fname = (
                    f"{lora_short}_"
                    f"cfg{cfg_v:.1f}_"
                    f"steps{steps_v}_"
                    f"{counter:05d}.png"
                )

                # Embed metadata in PNG
                pnginfo = PngInfo()
                pnginfo.add_text("batch_debug_params", json.dumps(meta, ensure_ascii=False))
                if prompt is not None:
                    pnginfo.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
                if extra_pnginfo is not None:
                    for k, v in extra_pnginfo.items():
                        pnginfo.add_text(k, json.dumps(v, ensure_ascii=False))

                filepath = os.path.join(full_output_folder, fname)
                pil_img.save(filepath, pnginfo=pnginfo, compress_level=self.compress_level)

                # Record the filename in metadata for CSV
                if i < len(metadata_records):
                    metadata_records[i]["filename"] = fname

                results.append({
                    "filename": fname,
                    "subfolder": subfolder,
                    "type": self.type,
                })
                counter += 1

        # --- 2. Save labeled grid ---
        if save_grid:
            # Build labels from metadata
            labels = []
            for i in range(n):
                meta = metadata_records[i] if i < len(metadata_records) else {}
                labels.append(format_label(label_format, meta))

            grid_pil = build_labeled_grid(
                [images[i] for i in range(n)],
                labels,
                columns=columns,
                cell_width=cell_width,
                max_grid_dimension=max_grid_dimension,
                font_size=label_font_size,
            )

            grid_fname = f"grid_{counter:05d}.png"
            grid_path = os.path.join(full_output_folder, grid_fname)

            # Embed sweep summary in grid PNG
            grid_pnginfo = PngInfo()
            grid_pnginfo.add_text("batch_debug_sweep_info",
                                  json.dumps({
                                      "total_images": n,
                                      "columns": columns,
                                      "label_format": label_format,
                                  }, ensure_ascii=False))
            grid_pil.save(grid_path, pnginfo=grid_pnginfo, compress_level=self.compress_level)

            results.append({
                "filename": grid_fname,
                "subfolder": subfolder,
                "type": self.type,
            })
            counter += 1
            print(f"[BatchDebug] Grid saved: {grid_fname}")

        # --- 3. Export CSV ---
        if save_metadata_csv and metadata_records:
            csv_fname = f"metadata_{counter:05d}.csv"
            csv_path = os.path.join(full_output_folder, csv_fname)
            write_metadata_csv(csv_path, metadata_records)
            print(f"[BatchDebug] Metadata CSV saved: {csv_fname}")

        # --- Summary ---
        total_saved = len(results)
        print(f"[BatchDebug] Output complete: {total_saved} files saved to {full_output_folder}")

        return {"ui": {"images": results}}
