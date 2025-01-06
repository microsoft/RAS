# Copyright 2024 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Any, Dict, Optional
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from ras.utils import ras_manager

def ras_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict=True,
        ) -> torch.Tensor:
    """
        Modified Forward pass of LuminaNextDiT for RAS.

        Parameters:
            hidden_states (torch.Tensor): Input tensor of shape (N, C, H, W).
            timestep (torch.Tensor): Tensor of diffusion timesteps of shape (N,).
            encoder_hidden_states (torch.Tensor): Tensor of caption features of shape (N, D).
            encoder_mask (torch.Tensor): Tensor of caption masks of shape (N, L).
    """
    hidden_states, mask, img_size, image_rotary_emb = self.patch_embedder(hidden_states, image_rotary_emb)
    image_rotary_emb = image_rotary_emb.to(hidden_states.device)

    temb = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)

    encoder_mask = encoder_mask.bool()
    if ras_manager.MANAGER.sample_ratio < 1.0:
        if ras_manager.MANAGER.is_RAS_step:
            hidden_states = hidden_states[:, ras_manager.MANAGER.other_patchified_index]
            ras_manager.MANAGER.image_rotary_emb_skip = image_rotary_emb[:, ras_manager.MANAGER.other_patchified_index]
    for layer in self.layers:
        hidden_states = layer(
            hidden_states,
            mask,
            image_rotary_emb,
            encoder_hidden_states,
            encoder_mask,
            temb=temb,
            cross_attention_kwargs=cross_attention_kwargs,
        )

    hidden_states = self.norm_out(hidden_states, temb)

    # unpatchify
    height_tokens = width_tokens = self.patch_size
    height, width = img_size[0]
    batch_size = hidden_states.size(0)
    sequence_length = (height // height_tokens) * (width // width_tokens)

    if ras_manager.MANAGER.sample_ratio < 1.0:
        if ras_manager.MANAGER.is_RAS_step:
            final_hidden_states = torch.zeros((hidden_states.shape[0], (height // height_tokens)*(width // width_tokens), hidden_states.shape[2]), device=hidden_states.device, dtype=hidden_states.dtype)
            final_hidden_states[:, ras_manager.MANAGER.other_patchified_index] = hidden_states
            hidden_states = final_hidden_states
    hidden_states = hidden_states[:, :sequence_length].view(
        batch_size, height // height_tokens, width // width_tokens, height_tokens, width_tokens, self.out_channels
    )
    output = hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
