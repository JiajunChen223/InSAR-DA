from __future__ import annotations

from insarda.config import MethodConfig
from insarda.methods.base import FormalMethod
from insarda.methods.source_only import SourceOnlyMethod
from insarda.methods.sft_replay import SFTReplayMethod
from insarda.methods.ss_coral import SSCORALMethod
from insarda.methods.ss_dann import SSDANNMethod
from insarda.methods.ss_mt import SSMTMethod
from insarda.methods.supervised_fine_tuning import SupervisedFineTuningMethod
from insarda.methods.st_joint import STJointMethod
from insarda.methods.target_only import TargetOnlyMethod


def build_method(
    name: str,
    method_cfg: MethodConfig,
    *,
    feature_dim: int | None = None,
) -> FormalMethod:
    method_name = str(name).strip().lower()
    if method_name == "source_only":
        return SourceOnlyMethod(method_cfg)
    if method_name == "target_only":
        return TargetOnlyMethod(method_cfg)
    if method_name == "supervised_fine_tuning":
        return SupervisedFineTuningMethod(method_cfg)
    if method_name == "sft_replay":
        return SFTReplayMethod(method_cfg)
    if method_name == "st_joint":
        return STJointMethod(method_cfg)
    if method_name == "ss_dann":
        if feature_dim is None:
            raise ValueError("`ss_dann` requires `feature_dim`.")
        return SSDANNMethod(feature_dim=feature_dim, method_cfg=method_cfg)
    if method_name == "ss_mt":
        return SSMTMethod(method_cfg)
    if method_name == "ss_coral":
        return SSCORALMethod(method_cfg)
    raise ValueError(f"Unknown method: {name}")


__all__ = [
    "FormalMethod",
    "SourceOnlyMethod",
    "TargetOnlyMethod",
    "SupervisedFineTuningMethod",
    "SFTReplayMethod",
    "STJointMethod",
    "SSDANNMethod",
    "SSMTMethod",
    "SSCORALMethod",
    "build_method",
]
