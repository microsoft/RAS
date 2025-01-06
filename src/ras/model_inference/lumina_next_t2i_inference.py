import argparse
import torch
from diffusers import LuminaText2ImgPipeline
from ras.utils.lumina_next_t2i.update_pipeline_lumina import update_lumina_pipeline
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args

def lumina_inf(args):
    pipeline = LuminaText2ImgPipeline.from_pretrained(
        "Alpha-VLLM/Lumina-Next-SFT-diffusers", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipeline = update_lumina_pipeline(pipeline)
    pipeline.transformer.to(memory_format=torch.channels_last)
    pipeline.vae.to(memory_format=torch.channels_last)
    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None
    numsteps = args.num_inference_steps
    image = pipeline(
                    generator=generator,
                    num_inference_steps=numsteps,
                    prompt=args.prompt,
                    height=args.height,
                    width=args.width,
                    ).images[0]
    image.save(args.output)


if __name__ == "__main__":
    args = parse_args()
    ras_manager.MANAGER.set_parameters(args)
    lumina_inf(args)
