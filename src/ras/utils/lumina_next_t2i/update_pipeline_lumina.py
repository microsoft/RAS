from ...modules.lumina_next_t2i.transformer_forward import ras_forward
from ...modules.attention_processor import RASLuminaAttnProcessor2_0
from ...schedulers import RASFlowMatchEulerDiscreteScheduler

def update_lumina_pipeline(pipeline):
    scheduler = RASFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = scheduler
    pipeline.transformer.forward = ras_forward.__get__(pipeline.transformer, pipeline.transformer.__class__)
    for block in pipeline.transformer.layers:
        block.attn1.set_processor(RASLuminaAttnProcessor2_0())
        block.attn2.set_processor(RASLuminaAttnProcessor2_0())
    return pipeline
