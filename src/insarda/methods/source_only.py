from __future__ import annotations

import torch.nn as nn

from insarda.methods.base import FormalMethod, LossOutput, masked_mse_loss


class SourceOnlyMethod(FormalMethod):
    name = "source_only"
    uses_target_labeled = False
    uses_target_unlabeled = False
    selection_loader_name = "source_val"

    def compute_loss(
        self,
        model: nn.Module,
        source_batch,
        target_labeled_batch,
        target_unlabeled_batch,
        ema_teacher,
        anchor_teacher,
    ) -> LossOutput:
        del target_labeled_batch, target_unlabeled_batch, ema_teacher, anchor_teacher
        if source_batch is None or source_batch["x"].shape[0] == 0:
            raise ValueError("`source_only` requires a non-empty `source_batch`.")
        source_prediction = model(source_batch["x"])
        source_loss = masked_mse_loss(source_prediction, source_batch["y"], source_batch["y_mask"])
        total = float(self.method_cfg.source_supervision_weight or 1.0) * source_loss
        return LossOutput(
            total=total,
            metrics={
                "source_loss": float(source_loss.detach().cpu().item()),
                "total_loss": float(total.detach().cpu().item()),
            },
        )
