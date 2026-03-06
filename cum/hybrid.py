import torch
from torch.optim.optimizer import Optimizer
from typing import List, Dict, Any, Optional
from .cum import CUM


class CUMWithAuxAdam(Optimizer):
    """
    Hybrid optimizer: CUM for 2D hidden weights, AdamW for everything else.

    Usage:
        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2 or p in embed_and_head]

        param_groups = [
            {"params": hidden_weights, "use_cum": True, "lr": 0.02},
            {"params": other_params, "use_cum": False, "lr": 3e-4, "betas": (0.9, 0.95)},
        ]
        optimizer = CUMWithAuxAdam(param_groups)
    """

    def __init__(self, param_groups: List[Dict[str, Any]], **cum_defaults):
        # Separate CUM and AdamW param groups
        cum_params = []
        adam_params = []

        for group in param_groups:
            use_cum = group.pop("use_cum", False)
            if use_cum:
                cum_params.append(group)
            else:
                adam_params.append(group)

        # Create internal optimizers
        cum_kwargs = {
            k: cum_defaults[k] for k in
            ["lr", "beta1", "beta2", "weight_decay", "ns_steps", "eps", "sigma_max", "alpha_damp", "nesterov"]
            if k in cum_defaults
        }

        if cum_params:
            first_cum = cum_params[0]
            cum_param_list = first_cum.pop("params")
            # Merge group-level overrides (excluding 'params') into cum_kwargs
            merged = {**cum_kwargs, **first_cum}
            self.cum_optimizer = CUM(cum_param_list, **merged)
        else:
            self.cum_optimizer = None

        # Handle multiple CUM groups
        for grp in cum_params[1:]:
            if self.cum_optimizer is not None:
                self.cum_optimizer.add_param_group(grp)

        adam_kwargs = {}
        if adam_params:
            adam_lr = adam_params[0].pop("lr", 3e-4)
            adam_betas = adam_params[0].pop("betas", (0.9, 0.95))
            adam_wd = adam_params[0].pop("weight_decay", 0.01)
            adam_kwargs = dict(lr=adam_lr, betas=adam_betas, weight_decay=adam_wd)

        self.adam_optimizer = torch.optim.AdamW(
            adam_params[0]["params"] if adam_params else [],
            **adam_kwargs,
        ) if adam_params else None

        for grp in adam_params[1:]:
            if self.adam_optimizer is not None:
                self.adam_optimizer.add_param_group(grp)

        # Combine all param groups for the parent class
        all_params = []
        if self.cum_optimizer:
            all_params.extend(self.cum_optimizer.param_groups)
        if self.adam_optimizer:
            all_params.extend(self.adam_optimizer.param_groups)

        # Initialize parent with dummy to avoid errors
        super().__init__([{"params": []}], {})
        self.param_groups = all_params

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.cum_optimizer is not None:
            self.cum_optimizer.step()
        if self.adam_optimizer is not None:
            self.adam_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        if self.cum_optimizer is not None:
            self.cum_optimizer.zero_grad(set_to_none=set_to_none)
        if self.adam_optimizer is not None:
            self.adam_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "cum": self.cum_optimizer.state_dict() if self.cum_optimizer else None,
            "adam": self.adam_optimizer.state_dict() if self.adam_optimizer else None,
        }

    def load_state_dict(self, state_dict):
        if self.cum_optimizer and state_dict.get("cum"):
            self.cum_optimizer.load_state_dict(state_dict["cum"])
        if self.adam_optimizer and state_dict.get("adam"):
            self.adam_optimizer.load_state_dict(state_dict["adam"])
