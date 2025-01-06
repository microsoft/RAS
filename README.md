<p align="center">
    <br>
    <img src="https://github.com/microsoft/RAS/blob/main/docs/img/logo_ras.png" width="400"/>
    <br>
<p>

# Region-Adaptive Sampling for Diffusion Transformers

**Authors**: [Ziming Liu](https://maruyamaaya.github.io/), [Yifan Yang](https://www.microsoft.com/en-us/research/people/yifanyang/), [Chengruidong Zhang](https://www.microsoft.com/en-us/research/people/chengzhang/), [Yiqi Zhang](https://viscent.dev/), [Lili Qiu](https://www.microsoft.com/en-us/research/people/liliqiu/), [Yang You](https://www.comp.nus.edu.sg/~youy/), [Yuqing Yang](https://www.microsoft.com/en-us/research/people/yuqyang/)

The RAS library is an open-source implementation of Regional-Adaptive Sampling (RAS), a novel diffusion model sampling strategy that introduces regional variability in sampling steps. Unlike conventional methods that uniformly process all image regions, RAS dynamically adjusts sampling ratios based on regional attention and noise metrics. This approach prioritizes computational resources for intricate regions while reusing previous outputs for less complex areas, achieving faster inference with minimal loss in image quality. Details can be found in our [research paper](arxiv.org/abs/xxxx.xxxxx).

<p align="center">
    <br>
    <img src="https://github.com/microsoft/RAS/blob/main/docs/img/showcase.png" width="800"/>
    <br>
<p>

Key features of the RAS library include:

- Dynamic Regional Sampling: Efficiently allocate computational power to regions requiring finer details.
- Training-Free Integration: Seamlessly integrate RAS with existing diffusion models like Stable Diffuion 3, Lumina-Next-T2I, and more.
- Flexible Tuning Space: Offer large tuning space of sample ratio, sample pattern, and so on, enabling flexible tradeoff between throughput and overall quality.
- User-Friendly API: Easily experiment with the DiT models by simply wrapping the pipeline with the RAS API.
Get started with RAS today to enhance your diffusion model's efficiency and unlock faster, high-quality generative capabilities!

<p align="center">
    <br>
    <img src="https://github.com/microsoft/RAS/blob/main/docs/img/drop_overview.png" width="600"/>
    <br>
<p>

## Installation

It is recommended to install RAS in a virtual environment. 
Notice that you need to install PyTorch according to your environment according to the [official documents](https://pytorch.org/).
```bash
conda create -n ras python=3.12
conda activate ras
git clone https://github.com/microsoft/RAS.git

install PyTorch according to your environment

cd RAS
python setup.py install
```

## Quickstart

Implementing RAS on [Diffusers](https://github.com/huggingface/diffusers) is easy. Here we provide two simple examples of RAS with the two models in our research paper. You can also modify the scripts for other usage.

**Stable Diffusion 3**

```python
import argparse
import torch
from diffusers import StableDiffusion3Pipeline
from ras.utils.Stable_Diffusion_3.update_pipeline_sd3 import update_sd3_pipeline
from ras.utils import ras_manager
from ras.utils.RAS_argparser import parse_args

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
```

```bash
cd scripts
bash Stable_Diffusion_3_example.sh
```

**Lumina Next T2I**

```python
import argparse
import torch
from diffusers import LuminaText2ImgPipeline
from ras.utils.Lumina_Next_T2I.update_pipeline_lumina import update_lumina_pipeline
from ras.utils import ras_manager
from ras.utils.RAS_argparser import parse_args

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

```


```bash
cd scripts
bash Lumina_Next_T2I_example.sh
```


## Customize Hyperparameters
**sample_ratio**: Average sample ratio for each RAS step. For instance, setting **sample_ratio** to 0.5 on a sequence of 4096 tokens will result in the noise of averagely 2048 tokens to be updated during each RAS step.

**replace_with_flash_attn**: Setting this will replace the attention kernel from torch.nn.functional.scaled_dot_product_attention to flash-attn. Please follow the [official document](https://github.com/Dao-AILab/flash-attention) to install flash-attn first.

**error_reset_steps**: The dense sampling steps inserted between the RAS steps to reset the accumulated error. Please use a string separated with commas for this parameter, such as "12,22".

**metric**: The metric used for identifying the importance of regions during the sampling process. Currently support "l2norm" and "std".

**high_ratio**: Based on the metric selected, the ratio of the high value chosen to be cached. Default value is set to 1.0, but can also be set between 0 and 1 to balance the sample ratio between the main subject and the background.

**starvation_scale**: RAS tracks how often a token is dropped and incorporate this count as a scaling factor in our metric for selecting tokens. This scale factor to prevent excessive blurring or noise in the final generated image. Larger scaleing factor will result in more uniform sampling. Usually set between 0 and 1.

**scheduler_start_step** and **scheduler_end_step**: set the range of sampling steps to apply RAS. The "scheduler_start_step" is recommended to be set to at least 4 to guarantee high generation quality.

**skip_num_step** and **skip_num_step_length**: The two parameters are set to enable linear dynamic sample ratio. The number of sampled tokens for each step increase/decrease by **skip_num_step** every **skip_num_step_length** steps.

**enable_index_fusion**: Whether to enable kernel fusion for higher generation speed. Please follow the [official document](https://wiki.tiker.net/PyCuda/Installation/) to install PyCuda first.

## Citation

```bibtex
@misc{liu2024regionadaptivesampling,
      title={Region-Adaptive Sampling for Diffusion Transformers}, 
      author={Ziming Liu and Yifan Yang and Chengruidong Zhang and Yiqi Zhang and Lili Qiu and Yang You and Yuqing Yang},
      year={2024},
      eprint={xxxx.xxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/xxxx.xxxxx}, 
}
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

