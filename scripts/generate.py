"""Text-to-image generation script.

Usage:
    python scripts/generate.py --config configs/generate.yaml
    python scripts/generate.py --config configs/generate.yaml --prompt "A cat in space"
"""

import argparse
import logging
import os
import sys

import torch
import yaml
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kamba.models.pipeline import LatentDiffusionModel
from kamba.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def main(config: dict) -> None:
    """Generate images from text prompts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    mc = config["model"]
    gc = config["generation"]
    cc = config.get("checkpoints", {})
    vae_cfg = mc.get("vae", {})
    den_cfg = mc.get("denoiser", {})

    # Load model
    logger.info("Loading LDM model...")
    model = LatentDiffusionModel(
        vae_config={
            "in_channels": vae_cfg.get("in_channels", 3),
            "latent_channels": vae_cfg.get("latent_channels", 4),
            "hidden_dims": tuple(vae_cfg.get("hidden_dims", [128, 256, 512, 512])),
            "image_size": gc.get("image_size", 256),
            "use_kan_decoder": vae_cfg.get("use_kan_decoder", True),
        },
        vae_checkpoint=cc.get("vae"),
        denoiser_config={
            "in_channels": den_cfg.get("in_channels", 4),
            "out_channels": den_cfg.get("out_channels", 4),
            "model_channels": den_cfg.get("model_channels", 320),
            "channel_mult": tuple(den_cfg.get("channel_mult", [1, 2, 4, 4])),
            "num_res_blocks": den_cfg.get("num_res_blocks", 2),
            "attention_resolutions": tuple(den_cfg.get("attention_resolutions", [1, 2, 3])),
            "context_dim": den_cfg.get("context_dim", 768),
            "num_heads": den_cfg.get("num_heads", 8),
            "use_cross_attn": den_cfg.get("use_cross_attn", True),
        },
        device=str(device),
    ).to(device)

    # Load denoiser checkpoint
    if cc.get("denoiser"):
        logger.info("Loading denoiser checkpoint: %s", cc["denoiser"])
        state_dict = torch.load(cc["denoiser"], map_location=device, weights_only=True)
        if "denoiser_state_dict" in state_dict:
            state_dict = state_dict["denoiser_state_dict"]
        model.denoiser.load_state_dict(state_dict)

    model.eval()

    # Prompts
    prompts = config.get("prompts", [
        "A photo of a cat sitting on a table",
        "A beautiful sunset over mountains",
    ])

    # Output directory
    output_dir = gc.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Generating %d images...", len(prompts))
    logger.info("Guidance scale: %s, Steps: %s", gc["guidance_scale"], gc["num_steps"])

    for idx, prompt in enumerate(prompts):
        logger.info("[%d/%d] %s", idx + 1, len(prompts), prompt)

        batch_prompts = [prompt] * gc.get("num_samples", 1)

        images = model.generate(
            captions=batch_prompts,
            num_steps=gc["num_steps"],
            guidance_scale=gc["guidance_scale"],
            height=gc.get("image_size", 256),
            width=gc.get("image_size", 256),
        )

        for i, img in enumerate(images):
            img = torch.clamp((img + 1) / 2, 0, 1)
            filename = f"{idx:03d}_{i}.png" if gc.get("num_samples", 1) > 1 else f"{idx:03d}.png"
            save_path = os.path.join(output_dir, filename)
            save_image(img, save_path)
            logger.info("  Saved: %s", save_path)

    logger.info(
        "Generated %d images in %s",
        len(prompts) * gc.get("num_samples", 1),
        output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--prompt", type=str, help="Single prompt (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.prompt:
        config["prompts"] = [args.prompt]
    if args.output_dir:
        config.setdefault("generation", {})["output_dir"] = args.output_dir

    setup_logger("kamba", ".")
    main(config)
