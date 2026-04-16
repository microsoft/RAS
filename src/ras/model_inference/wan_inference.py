import argparse
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
import sys

sys.path.append('/workspace/RAS')
sys.path.append('/workspace/RAS/src')
from ras.utils.wan.update_pipeline_wan import update_wan_pipeline
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args

def wan_inf(args):
    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipeline = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipeline.to("cuda")
    pipeline = update_wan_pipeline(pipeline)
    pipeline.enable_sequential_cpu_offload()
    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None
    numsteps = args.num_inference_steps
    video = pipeline(
        generator=generator,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=5.0,
        num_inference_steps=numsteps
    ).frames[0]
    export_to_video(video, args.output, fps=15)

if __name__ == "__main__":
    args = parse_args()
    ras_manager.MANAGER.set_parameters(args)
    wan_inf(args)
