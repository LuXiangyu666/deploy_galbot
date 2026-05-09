#!/usr/bin/env python3
"""Visualize Unitree G1 mocap trajectories in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import load_json_or_yaml
from g1_dance_controller import DanceController


ROOT = Path(__file__).resolve().parent
DEFAULT_MOCAP_PATH = ROOT / "storage" / "data" / "mocap" / "lafan1" / "UnitreeG1" / "dance1_subject3.npz"
DEFAULT_SCENE_PATH = ROOT / "assets" / "unitree_g1" / "scene.xml"
DEFAULT_DANCE_CONFIG = ROOT / "models" / "unitree_g1_fsm" / "dance" / "dance.json"
TRACE_RGBA = np.array([0.2, 0.8, 1.0, 1.0], dtype=np.float32)
TRACE_SIZE = np.array([0.015, 0.0, 0.0], dtype=np.float32)
IDENTITY_MAT = np.eye(3, dtype=np.float32).reshape(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use MuJoCo to replay a Unitree G1 mocap trajectory in real time."
    )
    parser.add_argument(
        "--motion",
        type=Path,
        default=DEFAULT_MOCAP_PATH,
        help="Path to the source mocap .npz file.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE_PATH,
        help="MuJoCo scene XML used for visualization.",
    )
    parser.add_argument(
        "--dance-config",
        type=Path,
        default=DEFAULT_DANCE_CONFIG,
        help="Config file used to obtain the 29-DoF default qpos.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame to play.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="Frame index after the last frame to play. Use -1 for the full clip.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for subsampling the clip.",
    )
    parser.add_argument(
        "--trace-length",
        type=int,
        default=180,
        help="Number of recent pelvis positions shown as the trajectory tail.",
    )
    parser.add_argument(
        "--height-offset",
        type=float,
        default=0.0,
        help="Additional offset added to root z for all frames.",
    )
    parser.add_argument(
        "--follow-root",
        action="store_true",
        help="Update the camera look-at target to follow the pelvis.",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to loop playback after the clip ends.",
    )
    return parser.parse_args()


def load_default_qpos(config_path: Path) -> np.ndarray:
    cfg = load_json_or_yaml(config_path)
    default_qpos = np.asarray(cfg["DEFAULT_QPOS"], dtype=np.float32).reshape(-1)
    if default_qpos.size != 29:
        raise ValueError(f"expected 29 default joint values in {config_path}, got {default_qpos.size}")
    return default_qpos


def load_reference_frames(motion_path: Path, default_qpos: np.ndarray, height_offset: float) -> tuple[np.ndarray, float]:
    motion = np.load(motion_path, allow_pickle=True)
    qpos = np.asarray(motion["qpos"], dtype=np.float32)
    joint_names = [str(name) for name in np.asarray(motion["joint_names"]).reshape(-1)]
    fps = float(np.asarray(motion["frequency"]).item())

    if qpos.ndim != 2 or qpos.shape[1] < 7:
        raise ValueError(f"invalid mocap qpos shape in {motion_path}: {qpos.shape}")
    if not joint_names or joint_names[0] != "root":
        raise ValueError(f"expected joint_names[0] to be 'root' in {motion_path}")
    if fps <= 0.0:
        raise ValueError(f"invalid mocap frequency in {motion_path}: {fps}")

    joint_to_index = DanceController._joint_name_to_index()
    joint_qpos = np.repeat(default_qpos.reshape(1, -1), qpos.shape[0], axis=0)
    mocap_joint_names = joint_names[1:]
    expected_joint_count = qpos.shape[1] - 7
    if len(mocap_joint_names) != expected_joint_count:
        raise ValueError(
            f"mocap joint_names mismatch in {motion_path}: "
            f"{len(mocap_joint_names)} names for {expected_joint_count} joint columns"
        )

    for mocap_col, joint_name in enumerate(mocap_joint_names, start=7):
        target_index = joint_to_index.get(joint_name)
        if target_index is None:
            continue
        joint_qpos[:, target_index] = qpos[:, mocap_col]

    full_qpos = np.concatenate([qpos[:, :7], joint_qpos], axis=1).astype(np.float32, copy=False)
    full_qpos[:, 2] += float(height_offset)
    return full_qpos, fps


def slice_frames(frames: np.ndarray, start_frame: int, end_frame: int, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if start_frame < 0:
        raise ValueError("--start-frame must be non-negative")

    stop = None if end_frame < 0 else end_frame
    clipped = frames[start_frame:stop:stride]
    if clipped.size == 0:
        raise ValueError("no frames left after applying start/end/stride")
    return clipped


def update_trace_geoms(mujoco, viewer, positions: np.ndarray) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    if positions.size == 0 or scene.maxgeom == 0:
        return

    count = min(len(positions), int(scene.maxgeom))
    start = len(positions) - count
    for geom_id, pos in enumerate(positions[start:]):
        rgba = TRACE_RGBA.copy()
        rgba[3] = 0.2 + 0.8 * (geom_id + 1) / count
        mujoco.mjv_initGeom(
            scene.geoms[geom_id],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            TRACE_SIZE,
            pos,
            IDENTITY_MAT,
            rgba,
        )
    scene.ngeom = count


def run_viewer(args: argparse.Namespace) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing mujoco. Install it first, for example: pip install mujoco") from exc

    default_qpos = load_default_qpos(args.dance_config)
    frames, mocap_fps = load_reference_frames(args.motion, default_qpos, args.height_offset)
    frames = slice_frames(frames, args.start_frame, args.end_frame, args.stride)

    if args.speed <= 0.0:
        raise ValueError("--speed must be positive")
    frame_dt = float(args.stride) / (mocap_fps * float(args.speed))

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    if model.nq < frames.shape[1]:
        raise ValueError(
            f"model nq={model.nq} cannot hold the trajectory qpos with {frames.shape[1]} values"
        )

    data.qpos[: frames.shape[1]] = frames[0]
    mujoco.mj_forward(model, data)

    trace_positions: list[np.ndarray] = []
    trace_length = max(0, int(args.trace_length))
    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -20.0
        if args.follow_root:
            viewer.cam.lookat[:] = frames[0, :3]

        frame_index = 0
        next_frame_time = time.monotonic()

        while viewer.is_running():
            data.qpos[: frames.shape[1]] = frames[frame_index]
            mujoco.mj_forward(model, data)

            if trace_length > 0:
                trace_positions.append(data.xpos[pelvis_body_id].copy())
                if len(trace_positions) > trace_length:
                    trace_positions = trace_positions[-trace_length:]
                update_trace_geoms(mujoco, viewer, np.asarray(trace_positions, dtype=np.float32))

            if args.follow_root:
                viewer.cam.lookat[:] = data.xpos[pelvis_body_id]

            viewer.sync()
            frame_index += 1

            if frame_index >= len(frames):
                if not args.loop:
                    break
                frame_index = 0
                next_frame_time = time.monotonic()
                trace_positions.clear()
                continue

            next_frame_time += frame_dt
            sleep_s = next_frame_time - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_frame_time = time.monotonic()


def main() -> None:
    args = parse_args()
    run_viewer(args)


if __name__ == "__main__":
    main()
