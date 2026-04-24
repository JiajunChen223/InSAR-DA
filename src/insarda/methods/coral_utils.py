from __future__ import annotations

import torch


def stable_float(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    return tensor


def safe_scalar_loss(loss: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    safe_loss = torch.nan_to_num(stable_float(loss), nan=0.0, posinf=0.0, neginf=0.0)
    if safe_loss.ndim != 0:
        safe_loss = safe_loss.mean()
    if not torch.isfinite(safe_loss):
        return reference.new_zeros(())
    return safe_loss.to(device=reference.device, dtype=reference.dtype)


def augment_sequence(
    x: torch.Tensor,
    *,
    noise_std: float = 0.01,
    drop_prob: float = 0.03,
    scale_std: float = 0.02,
) -> torch.Tensor:
    if x.numel() == 0:
        return x
    noise = torch.randn_like(x) * float(noise_std)
    keep = (torch.rand_like(x) > float(drop_prob)).float()
    scale = 1.0 + torch.randn(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) * float(scale_std)
    jittered = x * scale + noise
    return jittered * keep + x * (1.0 - keep)


def source_feature_stats(source_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = source_features.detach().mean(dim=0, keepdim=True)
    std = source_features.detach().std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    return mean, std


def shift_distance(features: torch.Tensor, source_mean: torch.Tensor, source_std: torch.Tensor) -> torch.Tensor:
    safe_features = torch.nan_to_num(stable_float(features), nan=0.0, posinf=1e3, neginf=-1e3)
    safe_mean = torch.nan_to_num(stable_float(source_mean), nan=0.0, posinf=1e3, neginf=-1e3)
    safe_std = torch.nan_to_num(stable_float(source_std), nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-4)
    return (((safe_features - safe_mean) / safe_std).pow(2).mean(dim=1, keepdim=True)).sqrt()


def medium_shift_gate(shift: torch.Tensor, *, width: float = 0.7) -> torch.Tensor:
    if shift.numel() == 0:
        return shift
    normalized = shift / shift.mean().detach().clamp_min(1e-6)
    gate = torch.exp(-((normalized - 1.0) ** 2) / float(2.0 * width * width))
    return gate.clamp(min=0.05, max=1.0)


def conservative_shift_gate(
    shift: torch.Tensor,
    *,
    width: float = 0.78,
    high_shift_center: float = 1.75,
    high_shift_temperature: float = 0.18,
) -> torch.Tensor:
    if shift.numel() == 0:
        return shift
    normalized = shift / shift.mean().detach().clamp_min(1e-6)
    base_gate = torch.exp(-((normalized - 1.0) ** 2) / float(2.0 * width * width))
    extreme_shift_penalty = torch.sigmoid(
        (float(high_shift_center) - normalized) / float(high_shift_temperature)
    )
    gate = base_gate * (0.2 + 0.8 * extreme_shift_penalty)
    return gate.clamp(min=0.03, max=1.0)


def _normalize_sample_weights(features: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return features.new_ones((features.shape[0], 1))
    normalized = torch.nan_to_num(stable_float(weights), nan=0.0, posinf=1.0, neginf=0.0)
    if normalized.ndim == 1:
        normalized = normalized.unsqueeze(1)
    return normalized.to(device=features.device, dtype=features.dtype).clamp_min(0.0)


def weighted_coral_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if source_features.numel() == 0 or target_features.numel() == 0:
        reference = source_features if source_features.numel() > 0 else target_features
        return reference.new_zeros(())
    source = torch.nan_to_num(stable_float(source_features), nan=0.0, posinf=1e3, neginf=-1e3)
    target = torch.nan_to_num(stable_float(target_features), nan=0.0, posinf=1e3, neginf=-1e3)
    source_w = _normalize_sample_weights(source, source_weights)
    target_w = _normalize_sample_weights(target, target_weights)
    source_norm = source_w.sum().clamp_min(1.0)
    target_norm = target_w.sum().clamp_min(1.0)
    source_mean = (source_w * source).sum(dim=0, keepdim=True) / source_norm
    target_mean = (target_w * target).sum(dim=0, keepdim=True) / target_norm
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_cov = (source_w * source_centered).transpose(0, 1) @ source_centered / source_norm
    target_cov = (target_w * target_centered).transpose(0, 1) @ target_centered / target_norm
    feature_dim = max(int(source.shape[-1]), 1)
    return ((source_cov - target_cov) ** 2).sum() / float(4 * feature_dim * feature_dim)


def temporal_coral_loss(
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if source_tokens.numel() == 0 or target_tokens.numel() == 0:
        reference = source_tokens if source_tokens.numel() > 0 else target_tokens
        return reference.new_zeros(())
    if source_tokens.shape[1] < 2 or target_tokens.shape[1] < 2:
        return source_tokens.new_zeros(())
    source_deltas = source_tokens[:, 1:] - source_tokens[:, :-1]
    target_deltas = target_tokens[:, 1:] - target_tokens[:, :-1]
    source_steps = int(source_deltas.shape[1])
    target_steps = int(target_deltas.shape[1])
    source_flat = source_deltas.reshape(-1, source_deltas.shape[-1])
    target_flat = target_deltas.reshape(-1, target_deltas.shape[-1])
    source_w = None
    target_w = None
    if source_weights is not None:
        source_w = _normalize_sample_weights(source_deltas[:, 0], source_weights).expand(-1, source_steps).reshape(-1, 1)
    if target_weights is not None:
        target_w = _normalize_sample_weights(target_deltas[:, 0], target_weights).expand(-1, target_steps).reshape(-1, 1)
    return weighted_coral_loss(source_flat, target_flat, source_weights=source_w, target_weights=target_w)


__all__ = [
    "augment_sequence",
    "conservative_shift_gate",
    "medium_shift_gate",
    "safe_scalar_loss",
    "shift_distance",
    "source_feature_stats",
    "stable_float",
    "temporal_coral_loss",
    "weighted_coral_loss",
]
