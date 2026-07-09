import torch

def configure_optimizers(model, weight_decay, learning_rate, betas, eps):
    """Replicates the split_decay logic for AdamW."""
    decay = set()
    no_decay = set()
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pn.endswith("bias") or "norm" in pn or "embedding" in pn:
            no_decay.add(p)
        else:
            decay.add(p)

    param_groups = [
        {"params": list(decay), "weight_decay": weight_decay},
        {"params": list(no_decay), "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas, eps=eps)