"""
Reward v2.1 P2/P3: epoch-level curriculum for objective scale, SLA weights, and
internal-path gamma (L_internal masking in training objective only).

P3 gamma: first ``REWARD_V2_GAMMA_WARMUP_EPOCHS`` (default 2) epochs use γ=0 so
agents learn entry migration without topology tearing pain; then γ→1 before eval.
"""

import os

from core.reward import clear_reward_v2_runtime, set_reward_v2_runtime


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def curriculum_enabled():
    return _env_flag("REWARD_V2_CURRICULUM", "0")


def internal_gamma_curriculum_enabled():
    return _env_flag("REWARD_V2_INTERNAL_GAMMA", "0")


def _lerp(start, end, progress):
    return float(start) + (float(end) - float(start)) * float(progress)


def curriculum_progress(epoch, num_epochs, *, is_eval_epoch=False):
    """0 at first train epoch → 1 at final eval epoch."""
    num_epochs = max(1, int(num_epochs))
    epoch = max(0, int(epoch))
    if is_eval_epoch or epoch >= num_epochs - 1:
        return 1.0
    train_span = max(1, num_epochs - 1)
    if epoch <= 0:
        return 0.0
    return min(1.0, epoch / float(max(1, train_span - 1)))


def internal_path_gamma(epoch, num_epochs, *, is_eval_epoch=False):
    """
    γ for L_e2e = L_acc + γ·L_internal in **training objective** only.

    Eval / final epoch always γ=1.0.
    """
    if not internal_gamma_curriculum_enabled():
        return 1.0
    num_epochs = max(1, int(num_epochs))
    epoch = max(0, int(epoch))
    if is_eval_epoch or epoch >= num_epochs - 1:
        return 1.0
    warmup = max(0, int(os.environ.get("REWARD_V2_GAMMA_WARMUP_EPOCHS", "2")))
    if epoch < warmup:
        return 0.0
    ramp_end = max(warmup, num_epochs - 2)
    if epoch >= ramp_end:
        return 1.0
    progress = (epoch - warmup) / float(max(1, ramp_end - warmup))
    return min(1.0, float(progress))


def curriculum_params(epoch, num_epochs, *, is_eval_epoch=False):
    """Compute curriculum knobs without mutating global reward state."""
    progress = curriculum_progress(epoch, num_epochs, is_eval_epoch=is_eval_epoch)
    scale_start = float(os.environ.get("REWARD_V2_SCALE_START", "150000"))
    scale_end = float(os.environ.get("REWARD_V2_SCALE_END", "60000"))
    params = {
        "progress": progress,
        "objective_scale_ms": _lerp(scale_start, scale_end, progress),
        "sla_alpha_mult": _lerp(
            float(os.environ.get("REWARD_V2_ALPHA_START_MULT", "0.5")),
            float(os.environ.get("REWARD_V2_ALPHA_END_MULT", "1.0")),
            progress,
        ),
        "sla_beta_mult": _lerp(
            float(os.environ.get("REWARD_V2_BETA_START_MULT", "0.5")),
            float(os.environ.get("REWARD_V2_BETA_END_MULT", "1.0")),
            progress,
        ),
        "migration_lambda_mult": _lerp(
            float(os.environ.get("REWARD_V2_MIG_LAMBDA_START_MULT", "0.6")),
            float(os.environ.get("REWARD_V2_MIG_LAMBDA_END_MULT", "1.25")),
            progress,
        ),
        "internal_path_gamma": internal_path_gamma(
            epoch, num_epochs, is_eval_epoch=is_eval_epoch
        ),
    }
    return params


def apply_curriculum_for_epoch(epoch, num_epochs, *, is_eval_epoch=False):
    """Apply runtime reward overrides for the current MARL epoch."""
    if not curriculum_enabled() and not internal_gamma_curriculum_enabled():
        clear_reward_v2_runtime()
        return None

    params = curriculum_params(epoch, num_epochs, is_eval_epoch=is_eval_epoch)
    runtime = {"internal_path_gamma": params["internal_path_gamma"]}
    if curriculum_enabled():
        runtime.update(
            {
                "objective_scale_ms": params["objective_scale_ms"],
                "sla_alpha_mult": params["sla_alpha_mult"],
                "sla_beta_mult": params["sla_beta_mult"],
                "migration_lambda_mult": params["migration_lambda_mult"],
            }
        )
    set_reward_v2_runtime(**runtime)
    return params


def reset_curriculum():
    clear_reward_v2_runtime()
