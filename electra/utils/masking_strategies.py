import torch

def dynamic_masking(
        special_tokens_mask: torch.Tensor,
        mask_prob: float
) -> torch.Tensor:
    prob_matrix = torch.full(
        special_tokens_mask.shape,
        mask_prob,
        device=special_tokens_mask.device
    )
    prob_matrix.masked_fill_(special_tokens_mask, value=0)

    mask = torch.bernoulli(prob_matrix).bool()

    return mask

def get_masking_strategy(masking_strategy: str):
    # Improve this
    if masking_strategy == 'dynamic_masking':
        return dynamic_masking
