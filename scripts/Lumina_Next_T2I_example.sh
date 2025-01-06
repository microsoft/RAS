
python ../src/ras/model_inference/lumina_next_t2i_inference.py \
    --prompt "A red heart in the clouds over water, in the style of zbrush, light pink and sky-blue, I can't believe how beautiful this is, hyperbolic expression, nyc explosion coverage, unreal engine 5, robert bissell." \
    --output "output.png" \
    --num_inference_steps 30 \
    --seed 42 \
    --sample_ratio 0.5 \
    --replace_with_flash_attn \
    --error_reset_steps "12,22" \
    --metric "std" \
    --scheduler_start_step 4 \
    --scheduler_end_step 30 \
    --patch_size 2 \
    --skip_num_step 256 \
    --skip_num_step_length 4 \
    # --enable_index_fusion \
