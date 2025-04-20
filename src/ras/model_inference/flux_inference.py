import argparse
import torch
import sys
import time

sys.path.append('/workspace/RAS')
sys.path.append('/workspace/RAS/src')
from diffusers import FluxPipeline
from ras.utils.flux.update_pipeline_flux import update_flux_pipeline
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args

def flux_inf(args):
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipeline = update_flux_pipeline(pipeline)
    pipeline.transformer.to(memory_format=torch.channels_last)
    pipeline.vae.to(memory_format=torch.channels_last)
    print("Vae", pipeline.vae_scale_factor)
    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None
    numsteps = args.num_inference_steps
    start = time.time()
    image = pipeline(
                    generator=generator,
                    num_inference_steps=numsteps,
                    prompt=args.prompt,
                    height=args.height,
                    width=args.width,
                    ).images[0]
    print(f"Pipeline time {time.time()-start}")
    image.save(args.output)


if __name__ == "__main__":
    args = parse_args()
    ras_manager.MANAGER.set_parameters(args)
    flux_inf(args)
