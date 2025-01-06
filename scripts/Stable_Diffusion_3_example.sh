
python ../src/ras/model_inference/stable_diffusion_3_inference.py \
    --prompt "A red heart in the clouds over water, in the style of zbrush, light pink and sky-blue, I can't believe how beautiful this is, hyperbolic expression, nyc explosion coverage, unreal engine 5, robert bissell." \
    --output "output.png" \
    --num_inference_steps 28 \
    --seed 29 \
    --sample_ratio 0.5 \
    --replace_with_flash_attn \
    --error_reset_steps "12,20" \
    --metric "std" \
    --scheduler_start_step 4 \
    --scheduler_end_step 28 \
    --patch_size 2 \
    --starvation_scale 1 \
    --high_ratio 0.3 \
    # --enable_index_fusion \
    # --skip_num_step 256 \
    # --skip_num_step_length 4 \
