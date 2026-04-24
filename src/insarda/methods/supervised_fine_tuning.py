from __future__ import annotations

import torch.nn as nn

from insarda.methods.base import FormalMethod, LossOutput, masked_mse_loss


class SupervisedFineTuningMethod(FormalMethod):
    name = "supervised_fine_tuning"
    uses_source_pretraining = True
    post_pretrain_uses_source = False
    post_pretrain_uses_target_labeled = True
    post_pretrain_uses_target_unlabeled = False
    uses_target_labeled = True
    selection_loader_name = "target_val"
    selection_smoothing_window = 3

    def __init__(self, method_cfg, **kwargs) -> None:
        del kwargs
        super().__init__(method_cfg)
        self._completed_epochs = 0
        pretrain_epochs = max(int(getattr(method_cfg, "source_pretrain_epochs", 0)), 0)
        self.source_pretrain_epochs = pretrain_epochs
        self.min_epochs_before_selection = max(pretrain_epochs + 1, 1)
        self.min_epochs_before_early_stop = max(pretrain_epochs + 1, 1)

    def on_epoch_end(self, *, epoch: int, epoch_record: dict[str, object]) -> dict[str, object]:
        del epoch_record
        self._completed_epochs = int(epoch)
        return {}

    def compute_loss(
        self,
        model: nn.Module,
        source_batch,
        target_labeled_batch,
        target_unlabeled_batch,
        ema_teacher,
        anchor_teacher,
    ):
        del target_unlabeled_batch, ema_teacher, anchor_teacher
        current_epoch = int(self._completed_epochs) + 1
        source_loss = None
        target_supervision_loss = None

        if current_epoch <= int(self.source_pretrain_epochs):
            if source_batch is None or source_batch["x"].shape[0] == 0:
                raise ValueError("`supervised_fine_tuning` source pretrain requires a non-empty `source_batch`.")
            source_prediction = model(source_batch["x"])
            source_loss = masked_mse_loss(source_prediction, source_batch["y"], source_batch["y_mask"])
            total = float(self.method_cfg.source_supervision_weight) * source_loss
            phase = "source_pretrain"
        else:
            if target_labeled_batch is None or target_labeled_batch["x"].shape[0] == 0:
                raise ValueError(
                    "`supervised_fine_tuning` target fine-tune phase requires a non-empty `target_labeled_batch`."
                )
            target_prediction = model(target_labeled_batch["x"])
            target_supervision_loss = masked_mse_loss(
                target_prediction,
                target_labeled_batch["y"],
                target_labeled_batch["y_mask"],
            )
            total = float(self.method_cfg.target_supervision_weight) * target_supervision_loss
            phase = "target_fine_tuning"

        if source_loss is None:
            source_loss = total.new_zeros(())
        if target_supervision_loss is None:
            target_supervision_loss = total.new_zeros(())

        return LossOutput(
            total=total,
            metrics={
                "source_loss": float(source_loss.detach().cpu().item()),
                "target_supervision_loss": float(target_supervision_loss.detach().cpu().item()),
                "total_loss": float(total.detach().cpu().item()),
                "training_phase": 0.0 if phase == "source_pretrain" else 1.0,
                "source_pretrain_epochs": float(self.source_pretrain_epochs),
            },
        )
