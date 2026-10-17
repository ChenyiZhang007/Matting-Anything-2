"""Memory-Separable Siamese mechanism (paper Section 3.5).

The same PDD is called twice. The mask pass receives memory-conditioned features;
this second, trimap pass receives the preserved features before memory attention.
The mechanism itself has no learned parameters and never registers another PDD.
"""

import torch
from torch import nn


class MemorySeparableSiamese(nn.Module):
    """Use the first pass's mask as a pseudo-prompt for the shared PDD."""

    def forward(
        self,
        decoder,
        prompt_encoder,
        image_features,
        mask_logits,
        high_res_features,
        multimask_output,
    ):
        batch = mask_logits.shape[0]
        coordinates = mask_logits.new_zeros(batch, 1, 2)
        labels = torch.full((batch, 1), -1, dtype=torch.int32, device=mask_logits.device)
        sparse, dense = prompt_encoder(points=(coordinates, labels), boxes=None, masks=mask_logits)
        trimap, *_ = decoder(
            image_embeddings=image_features,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return trimap
