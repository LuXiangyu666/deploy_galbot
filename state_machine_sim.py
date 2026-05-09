#!/usr/bin/env python3
"""MuJoCo preflight runner for the G1 dance policy.

The dance controller is the same class used by state_machine.py. This script
only replaces DDS lowstate with MuJoCo sensor/state reads and replaces DDS motor
commands with MuJoCo torque control.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from common import CONTROL_DT, G1_NUM_MOTOR, RobotState
from g1_dance_controller import DanceController


ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = ROOT / "models" / "unitree_g1_fsm"
DEFAULT_SCENE = Path("/home/galbot/Project/OpenTrack/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml")
DEFAULT_DANCE_REFERENCE = ROOT / "storage" / "data" / "mocap" / "lafan1" / "UnitreeG1" / "dance1_subject3.npz"
TORQUE_LIMIT = np.array(
    [
        88.0,
        139.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        139.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        50.0,
        50.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        5.0,
        5.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        5.0,
        5.0,
    ],
    dtype=np.float32,
)

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="MuJoCo scene XML.")
    parser.add_argument("--dt", type=float, default=CONTROL_DT, help="policy/control period in seconds")
    parser.add_argument("--sim-dt", type=float, default=0.002, help="MuJoCo integration timestep")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means run until closed/reference ends")
    parser.add_argument("--providers", default="CPUExecutionProvider", help="comma-separated ONNX Runtime providers")
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True, help="show MuJoCo viewer")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True, help="pace by wall clock")
    parser.add_argument("--print-every", type=int, default=50, help="print status every N policy ticks; 0 disables")

    parser.add_argument("--dance-policy", type=Path, default=DEFAULT_MODELS / "dance" / "model.onnx")
    parser.add_argument("--dance-config", type=Path, default=DEFAULT_MODELS / "dance" / "dance.json")
    parser.add_argument("--dance-reference", type=Path, default=DEFAULT_DANCE_REFERENCE)
    return parser.parse_args()


def quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in np.asarray(quat, dtype=np.float32).reshape(-1)[:4]]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def sensor_data(mujoco, model, data, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"MuJoCo model is missing sensor {name!r}")
    adr = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    return np.asarray(data.sensordata[adr : adr + dim], dtype=np.float32).copy()


def validate_model_layout(mujoco, model) -> None:
    if model.nq < 7 + G1_NUM_MOTOR or model.nv < 6 + G1_NUM_MOTOR:
        raise ValueError(f"scene must expose free root + 29 joints, got nq={model.nq}, nv={model.nv}")
    if model.nu != G1_NUM_MOTOR:
        raise ValueError(f"scene must expose 29 torque actuators, got nu={model.nu}")

    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
        for idx in range(model.nu)
    ]
    if actuator_names != JOINT_NAMES:
        raise ValueError("MuJoCo actuator order does not match G1 controller joint order")

    for sensor_name in ("gyro_pelvis", "orientation_pelvis"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) < 0:
            raise ValueError(f"scene must include sensor {sensor_name!r}")


def robot_state_from_mujoco(mujoco, model, data) -> RobotState:
    quat = sensor_data(mujoco, model, data, "orientation_pelvis")
    return RobotState(
        qpos=np.asarray(data.qpos[7 : 7 + G1_NUM_MOTOR], dtype=np.float32).copy(),
        qvel=np.asarray(data.qvel[6 : 6 + G1_NUM_MOTOR], dtype=np.float32).copy(),
        quat=quat,
        rpy=quat_wxyz_to_rpy(quat),
        gyro=sensor_data(mujoco, model, data, "gyro_pelvis"),
        mode_machine=0,
        wireless_remote=b"",
    )


def apply_pd_torque(model, data, target: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
    qpos = np.asarray(data.qpos[7 : 7 + G1_NUM_MOTOR], dtype=np.float32)
    qvel = np.asarray(data.qvel[6 : 6 + G1_NUM_MOTOR], dtype=np.float32)
    torque = kp * (target - qpos) - kd * qvel
    torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)
    data.ctrl[:] = torque


def initialize_from_reference(mujoco, model, data, dance: DanceController, row: int = 0) -> None:
    data.qpos[:] = np.concatenate(
        [
            dance.ref_root_pos_all[row],
            dance.ref_root_quat_all[row],
            dance.ref_qpos_all[row],
        ]
    ).astype(np.float32, copy=False)
    data.qvel[:] = np.concatenate(
        [
            dance.ref_root_qvel_all[row],
            dance.ref_joint_vel_all[row],
        ]
    ).astype(np.float32, copy=False)
    data.ctrl[:] = data.qpos[7 : 7 + G1_NUM_MOTOR]
    mujoco.mj_forward(model, data)


def command_from_key(command: dict[str, bool], key: int) -> None:
    if key in (256,):  # GLFW_KEY_ESCAPE
        command["quit"] = True
        return
    if not 0 <= key < 256:
        return

    char = chr(key).lower()
    if char == "r":
        command["reset"] = True


def print_controls() -> None:
    print("Keyboard: R reset dance test, Esc quit")


def run_sim(args: argparse.Namespace) -> None:
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing mujoco. Install it before running state_machine_sim.py") from exc

    providers = args.providers.split(",") if args.providers else None
    dance = DanceController(
        args.dance_policy,
        args.dance_config,
        providers,
        reference_path=args.dance_reference,
        control_dt=args.dt,
    )

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    model.opt.timestep = float(args.sim_dt)
    data = mujoco.MjData(model)
    validate_model_layout(mujoco, model)

    initialize_from_reference(mujoco, model, data, dance, row=0)
    state = robot_state_from_mujoco(mujoco, model, data)
    dance.reset(state.qpos)

    command = {"quit": False, "reset": False}
    substeps = max(1, int(float(args.dt) / float(args.sim_dt)))
    max_steps = dance.end_iter if args.duration_s <= 0.0 else max(1, int(round(args.duration_s / args.dt)))

    print(
        f"SIM DANCE: using the same DanceController as state_machine.py; "
        f"dance_obs={dance.tracking_input_size}, steps={max_steps}, substeps={substeps}"
    )
    print_controls()

    def reset_dance() -> None:
        initialize_from_reference(mujoco, model, data, dance, row=0)
        state = robot_state_from_mujoco(mujoco, model, data)
        dance.reset(state.qpos)
        command["reset"] = False
        print("RESET -> DANCE")

    def tick(step: int) -> bool:
        if command["reset"]:
            reset_dance()
        current_state = robot_state_from_mujoco(mujoco, model, data)
        target, kp, kd = dance.calculate(current_state), dance.kp, dance.kd
        if dance.done:
            return False

        for _ in range(substeps):
            apply_pd_torque(model, data, target, kp, kd)
            mujoco.mj_step(model, data)

        if args.print_every > 0 and step % args.print_every == 0:
            err = float(np.linalg.norm(target - data.qpos[7 : 7 + G1_NUM_MOTOR]))
            print(
                f"step={step:05d} mode=dance time={data.time:7.3f} "
                f"frame={dance.inference_counter:05d}/{dance.end_iter:05d} target_err={err:.4f}"
            )
        return True

    if args.viewer:
        try:
            import mujoco.viewer
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing mujoco.viewer support in this Python environment") from exc

        with mujoco.viewer.launch_passive(model, data, key_callback=lambda key: command_from_key(command, key)) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 140.0
            viewer.cam.elevation = -20.0
            viewer.cam.lookat[:] = data.qpos[:3]
            next_tick = time.monotonic()
            for step in range(max_steps):
                if command["quit"] or not viewer.is_running():
                    break
                if not tick(step):
                    break
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
                if args.real_time:
                    next_tick += float(args.dt)
                    sleep_s = next_tick - time.monotonic()
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                    else:
                        next_tick = time.monotonic()
    else:
        for step in range(max_steps):
            if command["quit"]:
                break
            if not tick(step):
                break

    print(f"SIM DANCE: finished policy_time={data.time:.3f} frame={dance.inference_counter}/{dance.end_iter}")


def main() -> None:
    args = parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
