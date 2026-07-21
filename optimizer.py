import torch

def configure_adamw(model, weight_decay, learning_rate, betas, eps):
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

def configure_sgd(model, learning_rate, momentum):
    """Configures standard SGD with Momentum."""
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Utilizes a quintic iteration to maximize the slope at zero.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # bfloat16 delivers extremely fast matrix mults if supported
    if G.dtype == torch.bfloat16 or (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        X = G.bfloat16()
    else:
        X = G.float()

    if X.size(0) > X.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + eps)
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        
    if G.size(0) > G.size(1):
        X = X.T
        
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                    
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                    
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)
                
                if group['nesterov']:
                    g = g.add(buf, alpha=group['momentum'])
                else:
                    g = buf.clone()
                
                # Orthogonalize the gradient
                g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                
                # Scale update based on the aspect ratio properties natively via Muon logic
                scale = max(1.0, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-group['lr'] * scale)
                
        return loss

class MultiOptimizer:
    """Wrapper to safely handle step() & zero_grad() for multiple independent optimizers"""
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()

def configure_muon(model, muon_lr=0.02, muon_momentum=0.95, adamw_lr=1e-4, adamw_betas=(0.9, 0.95), adamw_eps=1e-8, adamw_wd=0.01):
    """
    Muon specifically optimizes >= 2D matrices (hidden weights).
    1D weights (biases, embeddings, norms) and the LM Head fallback to AdamW.
    """
    muon_params = []
    adamw_decay_params = []
    adamw_nodecay_params = []
    
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # >= 2D params in the transformer body map to Muon
        if p.ndim >= 2 and 'embedding' not in pn and 'lm_head' not in pn:
            muon_params.append(p)
        else:
            # The rest fall back to AdamW
            if pn.endswith("bias") or "norm" in pn or "embedding" in pn:
                adamw_nodecay_params.append(p)
            else:
                adamw_decay_params.append(p)
                
    muon_opt = Muon(muon_params, lr=muon_lr, momentum=muon_momentum)
    
    adamw_groups = []
    if adamw_decay_params:
        adamw_groups.append({"params": adamw_decay_params, "weight_decay": adamw_wd})
    if adamw_nodecay_params:
        adamw_groups.append({"params": adamw_nodecay_params, "weight_decay": 0.0})
        
    adamw_opt = torch.optim.AdamW(adamw_groups, lr=adamw_lr, betas=adamw_betas, eps=adamw_eps)
    
    return MultiOptimizer(muon_opt, adamw_opt)
