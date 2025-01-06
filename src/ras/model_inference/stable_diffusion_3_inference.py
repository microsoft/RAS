import argparse
import torch
from diffusers import StableDiffusion3Pipeline
from ras.utils.stable_diffusion_3.update_pipeline_sd3 import update_sd3_pipeline
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args

def sd3_inf(args):
    pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, use_auth_token=True)
    pipeline.to("cuda")
    pipeline = update_sd3_pipeline(pipeline)
    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None
    numsteps = args.num_inference_steps
    image = pipeline(
                    generator=generator,
                    num_inference_steps=numsteps,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    guidance_scale=7.0,
                    ).images[0]
    image.save(args.output)

if __name__ == "__main__":
    args = parse_args()
    ras_manager.MANAGER.set_parameters(args)
    sd3_inf(args)
