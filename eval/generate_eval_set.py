"""Generate raw 512x512 images for every (test prompt x seed) for one model variant.

Run on a GPU (Colab).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform

from paintbypixel.config import load_config, read_prompts, resolve
from paintbypixel.generate import generate_image, load_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="name for this config, e.g. base, lora_r4, lora_r16")
    ap.add_argument("--lora", default=None, help="LoRA dir or HF repo id (omit for the base model)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prompts = read_prompts(cfg)[: args.limit]
    out_dir = resolve(cfg["eval"]["out_dir"]) / "raw" / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    import diffusers
    import torch

    (out_dir / "run_config.json").write_text(json.dumps({
        "variant": args.variant, "lora": args.lora, "config": cfg,
        "torch": torch.__version__, "diffusers": diffusers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
    }, indent=2))

    pipe = load_pipeline(cfg, args.lora)
    manifest = out_dir / "manifest.csv"
    new_file = not manifest.exists()
    with open(manifest, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["file", "prompt_id", "category", "subject", "seed", "seconds", "nsfw_flag"])
        for p in prompts:
            for seed in cfg["generation"]["seeds"]:
                name = f"{p['id']:03d}_s{seed}.png"
                if (out_dir / name).exists():
                    continue
                img, info = generate_image(pipe, cfg, p["subject"], seed)
                img.save(out_dir / name)
                w.writerow([name, p["id"], p["category"], p["subject"], seed, f"{info['seconds']:.3f}", int(info["nsfw_flag"])])
                f.flush()
                print(f"[{args.variant}] {name} {p['subject']!r} {info['seconds']:.1f}s")
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
