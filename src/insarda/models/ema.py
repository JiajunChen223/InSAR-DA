from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMATeacher:
    def __init__(self, model: nn.Module, decay: float = 0.995, method: nn.Module | None = None) -> None:
        self.decay = float(decay)
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.method_teacher = copy.deepcopy(method) if method is not None else None
        if self.method_teacher is not None:
            self.method_teacher.eval()
            for parameter in self.method_teacher.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def _update_module(self, teacher_module: nn.Module, source_module: nn.Module) -> None:
        teacher_state = teacher_module.state_dict()
        source_state = source_module.state_dict()
        for key, value in teacher_state.items():
            source = source_state[key].detach()
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)
        teacher_module.load_state_dict(teacher_state, strict=True)
        teacher_module.eval()

    @torch.no_grad()
    def update(self, model: nn.Module, method: nn.Module | None = None) -> None:
        self._update_module(self.teacher, model)
        if self.method_teacher is not None and method is not None:
            self._update_module(self.method_teacher, method)

    def __call__(self, *args, **kwargs):
        return self.teacher(*args, **kwargs)

    def __getattr__(self, name: str):
        if name in {"teacher", "method_teacher", "decay"}:
            raise AttributeError(name)
        return getattr(self.teacher, name)
