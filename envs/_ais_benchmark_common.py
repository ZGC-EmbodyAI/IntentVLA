from .utils import *
import sapien
import numpy as np
import os


PAD_COLOR_POOL = [
    (0.12, 0.58, 0.95),
    (0.24, 0.78, 0.36),
    (0.72, 0.32, 0.92),
    (0.95, 0.22, 0.18),
    (0.95, 0.86, 0.16),
    (0.95, 0.55, 0.12),
    (0.95, 0.28, 0.52),
    (0.10, 0.75, 0.72),
]


OBJECT_COLOR_POOL = [
    ("blue", (0.12, 0.58, 0.95)),
    ("green", (0.24, 0.78, 0.36)),
    ("purple", (0.72, 0.32, 0.92)),
    ("red", (0.95, 0.22, 0.18)),
    ("yellow", (0.95, 0.86, 0.16)),
    ("orange", (0.95, 0.55, 0.12)),
    ("cyan", (0.10, 0.75, 0.72)),
    ("pink", (0.95, 0.35, 0.70)),
]




def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def use_fixed_aliasbench_colors() -> bool:
    return _env_flag("AISBENCH_FIXED_COLORS", False)


def fixed_pad_colors(num):
    return PAD_COLOR_POOL[:num]


def fixed_block_colors(num):
    return OBJECT_COLOR_POOL[:num]

def sample_pad_color():
    sampled_idx = np.random.randint(0, len(PAD_COLOR_POOL))
    if use_fixed_aliasbench_colors():
        return PAD_COLOR_POOL[0]
    return PAD_COLOR_POOL[sampled_idx]


def sample_distinct_pad_colors(num):
    ids = np.random.choice(len(PAD_COLOR_POOL), num, replace=False)
    if use_fixed_aliasbench_colors():
        return fixed_pad_colors(num)
    return [PAD_COLOR_POOL[int(color_id)] for color_id in ids]


def sample_block_colors(num):
    ids = np.random.choice(len(OBJECT_COLOR_POOL), num, replace=False)
    if use_fixed_aliasbench_colors():
        return fixed_block_colors(num)
    return [OBJECT_COLOR_POOL[int(color_id)] for color_id in ids]


def create_visual_pad(scene, center, half_size, color, name):
    return create_visual_box(
        scene=scene,
        pose=sapien.Pose(center.tolist(), [1, 0, 0, 0]),
        half_size=half_size,
        color=color,
        name=name,
    )


def mirrored_arm_and_sign():
    arm_tag = ArmTag("left" if np.random.randint(0, 2) == 0 else "right")
    x_sign = -1 if arm_tag == "left" else 1
    return arm_tag, x_sign


def opposite_centers(x_sign, z=0.742):
    return {
        "A": np.array([x_sign * 0.02, -0.14, z]),
        "B": np.array([x_sign * 0.24, -0.14, z]),
        "C": np.array([x_sign * 0.02, 0.00, z]),
        "D": np.array([x_sign * 0.24, 0.00, z]),
    }


def opposite_map():
    return {"A": "D", "D": "A", "B": "C", "C": "B"}


def target_pose_from_center(center, table_z_bias=0.0, quat=None):
    target_center = np.array(center, dtype=float).copy()
    target_center[2] += table_z_bias
    if quat is None:
        quat = [0, 1, 0, 0]
    return target_center.tolist() + list(quat)


def near_xy(actor_pos, target_pos, eps_xy):
    return np.all(np.abs(np.array(actor_pos[:2]) - np.array(target_pos[:2])) < np.array(eps_xy))
