"""Stage 1 — text to image with Stable Diffusion 1.5 (+ optional LoRA). Needs a GPU."""
from __future__ import annotations

import time

from .config import build_prompt

SCHEDULERS = {
    "dpmpp": "DPMSolverMultistepScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    "ddim": "DDIMScheduler",
}


def load_pipeline(cfg: dict, lora_path: str | None = None, device: str | None = None):
    import diffusers
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(cfg["model"]["base_model"], torch_dtype=dtype)
    sched_cls = getattr(diffusers, SCHEDULERS[cfg["generation"]["scheduler"]])
    pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)
    if lora_path:
        pipe.load_lora_weights(lora_path)
    pipe._pbp_has_lora = bool(lora_path)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def generate_image(pipe, cfg: dict, subject: str, seed: int, lora_scale: float | None = None):
    """Generate one 512x512 image. Same (subject, seed, cfg) -> same image."""
    import torch

    g = cfg["generation"]
    generator = torch.Generator(device=pipe.device.type).manual_seed(int(seed))
    kwargs = {}
    if getattr(pipe, "_pbp_has_lora", False):
        kwargs["cross_attention_kwargs"] = {"scale": lora_scale if lora_scale is not None else cfg["model"]["lora_scale"]}
    t0 = time.perf_counter()
    out = pipe(
        prompt=build_prompt(cfg, subject),
        negative_prompt=g["negative_prompt"],
        num_inference_steps=g["num_inference_steps"],
        guidance_scale=g["guidance_scale"],
        height=g["height"],
        width=g["width"],
        generator=generator,
        **kwargs,
    )
    if pipe.device.type == "cuda":
        torch.cuda.synchronize()
    nsfw = getattr(out, "nsfw_content_detected", None)
    info = {"seconds": time.perf_counter() - t0, "nsfw_flag": bool(nsfw[0]) if nsfw else False}
    return out.images[0], info
