from __future__ import annotations

import torch.nn as nn

from insarda.methods.base import FormalMethod, LossOutput, masked_mse_loss


class TargetOnlyMethod(FormalMethod):
    name = "target_only"
    uses_source = False
    uses_target_labeled = True
    selection_loader_name = "target_val"

    def compute_loss(
        self,
        model: nn.Module,
        source_batch,
        target_labeled_batch,
        target_unlabeled_batch,
        ema_teacher,
        anchor_teacher,
    ):
        del source_batch, target_unlabeled_batch, ema_teacher, anchor_teacher
        if target_labeled_batch is None or target_labeled_batch["x"].shape[0] == 0:
            raise ValueError("`target_only` requires a non-empty `target_labeled_batch`.")
        target_prediction = model(target_labeled_batch["x"])
        target_supervision_loss = masked_mse_loss(
            target_prediction,
            target_labeled_batch["y"],
            target_labeled_batch["y_mask"],
        )
        total = float(self.method_cfg.target_supervision_weight or 1.0) * target_supervision_loss
        return LossOutput(
            total=total,
            metrics={
                "target_supervision_loss": float(target_supervision_loss.detach().cpu().item()),
                "total_loss": float(total.detach().cpu().item()),
            },
        )
