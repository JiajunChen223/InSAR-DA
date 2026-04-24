from __future__ import annotations

import torch.nn as nn

from insarda.methods.base import LossOutput, masked_mse_loss
from insarda.methods.supervised_fine_tuning import SupervisedFineTuningMethod


class SFTReplayMethod(SupervisedFineTuningMethod):
    name = "sft_replay"
    post_pretrain_uses_source = True

    def _replay_weight(self, current_epoch: int) -> float:
        target_stage_epoch = max(int(current_epoch) - int(self.source_pretrain_epochs), 1)
        base_weight = float(getattr(self.method_cfg, "replay_source_weight", 0.0))
        return base_weight / (1.0 + 0.05 * float(max(target_stage_epoch - 1, 0)))

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
        replay_source_loss = None

        if current_epoch <= int(self.source_pretrain_epochs):
            if source_batch is None or source_batch["x"].shape[0] == 0:
                raise ValueError("`sft_replay` source pretrain requires a non-empty `source_batch`.")
            source_prediction = model(source_batch["x"])
            source_loss = masked_mse_loss(source_prediction, source_batch["y"], source_batch["y_mask"])
            total = float(self.method_cfg.source_supervision_weight) * source_loss
            phase = "source_pretrain"
            replay_weight = 0.0
        else:
            if source_batch is None or source_batch["x"].shape[0] == 0:
                raise ValueError("`sft_replay` target fine-tune phase requires a non-empty `source_batch`.")
            if target_labeled_batch is None or target_labeled_batch["x"].shape[0] == 0:
                raise ValueError("`sft_replay` target fine-tune phase requires a non-empty `target_labeled_batch`.")
            source_replay_prediction = model(source_batch["x"])
            replay_source_loss = masked_mse_loss(source_replay_prediction, source_batch["y"], source_batch["y_mask"])
            target_prediction = model(target_labeled_batch["x"])
            target_supervision_loss = masked_mse_loss(
                target_prediction,
                target_labeled_batch["y"],
                target_labeled_batch["y_mask"],
            )
            replay_weight = self._replay_weight(current_epoch)
            total = (
                float(self.method_cfg.target_supervision_weight) * target_supervision_loss
                + replay_weight * replay_source_loss
            )
            phase = "target_fine_tuning"

        if source_loss is None:
            source_loss = total.new_zeros(())
        if target_supervision_loss is None:
            target_supervision_loss = total.new_zeros(())
        if replay_source_loss is None:
            replay_source_loss = total.new_zeros(())

        return LossOutput(
            total=total,
            metrics={
                "source_loss": float(source_loss.detach().cpu().item()),
                "target_supervision_loss": float(target_supervision_loss.detach().cpu().item()),
                "replay_source_loss": float(replay_source_loss.detach().cpu().item()),
                "replay_source_weight": float(replay_weight),
                "training_phase": 0.0 if phase == "source_pretrain" else 1.0,
                "source_pretrain_epochs": float(self.source_pretrain_epochs),
                "total_loss": float(total.detach().cpu().item()),
            },
        )
