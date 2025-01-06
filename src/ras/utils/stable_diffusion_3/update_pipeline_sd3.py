from ...schedulers import RASFlowMatchEulerDiscreteScheduler
from ...modules.attention_processor import RASJointAttnProcessor2_0
from ...modules.stable_diffusion_3.transformer_forward import ras_forward

def update_sd3_pipeline(pipeline):
    scheduler = RASFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = scheduler
    pipeline.transformer.forward = ras_forward.__get__(pipeline.transformer, pipeline.transformer.__class__)
    for block in pipeline.transformer.transformer_blocks:
        block.attn.set_processor(RASJointAttnProcessor2_0())
    return pipeline
