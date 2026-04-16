from ...schedulers import RASFluxFlowMatchEulerDiscreteScheduler
from ...modules.attention_processor import RASFluxAttnProcessor2_0
from ...modules.flux.transformer_forward import ras_forward

def update_flux_pipeline(pipeline):
    scheduler = RASFluxFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = scheduler
    pipeline.transformer.forward = ras_forward.__get__(pipeline.transformer, pipeline.transformer.__class__)
    for block in pipeline.transformer.transformer_blocks:
        block.attn.set_processor(RASFluxAttnProcessor2_0())
    for block in pipeline.transformer.single_transformer_blocks:
        block.attn.set_processor(RASFluxAttnProcessor2_0())
    return pipeline