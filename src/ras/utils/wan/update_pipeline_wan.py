from ...schedulers import RASWanFlowMatchEulerDiscreteScheduler
from ...modules.attention_processor import RASWanAttnProcessor2_0
from ...modules.wan.transformer_forward import ras_forward

def update_wan_pipeline(pipeline):
    scheduler = RASWanFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = scheduler
    pipeline.transformer.forward = ras_forward.__get__(pipeline.transformer, pipeline.transformer.__class__)
    for block in pipeline.transformer.blocks:
        block.attn1.set_processor(RASWanAttnProcessor2_0())
        block.attn2.set_processor(RASWanAttnProcessor2_0())
    return pipeline