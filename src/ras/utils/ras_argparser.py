import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Lumina Inference")
    parser.add_argument("--prompt", type=str, default="A red heart in the clouds over water, in the style of zbrush, light pink and sky-blue, I can't believe how beautiful this is, hyperbolic expression, nyc explosion coverage, unreal engine 5, robert bissell.")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--output", type=str, default="output.png", help="Output file path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=0.5, help="Average sample ratio during each step for RAS")
    parser.add_argument("--replace_with_flash_attn", action="store_true", help="Replace the attention mechanism with flash attention")
    parser.add_argument("--error_reset_steps", type=str, default="12,22", help="Dense steps to reset the error, use comma to separate")
    parser.add_argument("--metric", type=str, default="std", choices=["std", "l2norm"], help="Metric to calculate the patch selection, currently support std and l2norm")
    parser.add_argument("--high_ratio", type=float, default=1, help="Based on the metric selected, the ratio of the high value chosen to be cached. Range from 0 to 1")
    parser.add_argument("--height", type=int, default=1024, help="Height of the image")
    parser.add_argument("--width", type=int, default=1024, help="Width of the image")
    parser.add_argument("--starvation_scale", type=float, default=0.1, help="Starvation scale for the patch selection, higher value will make the patch selection more average")
    parser.add_argument("--scheduler_start_step", type=int, default=4, help="From which sampling step to start using RAS")
    parser.add_argument("--scheduler_end_step", type=int, default=30, help="From which sampling step to stop using RAS")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size for Lumina")
    parser.add_argument("--skip_num_step", type=int, default=0, help="Parameter for dynamic skip token numbers. The skip token number increases/decreases by this value every skip_num_step_length steps. Can be negative or positive.")
    parser.add_argument("--skip_num_step_length", type=int, default=0, help="The interval to change the skip token number")
    parser.add_argument("--enable_index_fusion", action="store_true", help="Enable index fusion for RAS")

    parser.add_argument("--num_frames", type=int, default=81, help="Num frames for video generation models.")
    parser.add_argument("--temporal_patch_size", type=int, default=1, help="temporal patch size")
    parser.add_argument("--is_video", action="store_true", help="Whether current model is a video model")

    return parser.parse_args()
