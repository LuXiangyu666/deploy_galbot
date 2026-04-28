#!/usr/bin/env python3
"""Shared types and utilities for the G1 DDS state machine."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

G1_NUM_MOTOR = 29
MODE_PR = 0
ARM_IDS = np.arange(15, 29, dtype=np.int32)
CONTROL_DT = 0.02


def _field(obj: Any, name: str) -> Any:
    value = getattr(obj, name)
    return value() if callable(value) else value


class Button:
    def __init__(self) -> None:
        self.pressed = False
        self.on_press = False
        self.on_release = False

    def update(self, state: bool) -> None:
        state = bool(state)
        self.on_press = state and not self.pressed
        self.on_release = (not state) and self.pressed
        self.pressed = state


class Gamepad:
    BUTTON_BITS = {
        "R1": 0,
        "L1": 1,
        "start": 2,
        "select": 3,
        "R2": 4,
        "L2": 5,
        "F1": 6,
        "F2": 7,
        "A": 8,
        "B": 9,
        "X": 10,
        "Y": 11,
        "up": 12,
        "right": 13,
        "down": 14,
        "left": 15,
    }

    def __init__(self, smooth: float = 0.3, dead_zone: float = 0.01) -> None:
        self.smooth = float(smooth)
        self.dead_zone = float(dead_zone)

        self.lx = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.l2 = 0.0
        self.ly = 0.0

        for name in self.BUTTON_BITS:
            setattr(self, name, Button())

    def _axis(self, previous: float, raw: float) -> float:
        value = 0.0 if abs(raw) < self.dead_zone else raw
        return previous * (1.0 - self.smooth) + value * self.smooth

    def update(self, wireless_remote: bytes | bytearray | list[int] | tuple[int, ...]) -> None:
        data = bytes(int(v) & 0xFF for v in wireless_remote)
        if len(data) < 24:
            return

        buttons = int.from_bytes(data[2:4], byteorder="little", signed=False)
        for name, bit in self.BUTTON_BITS.items():
            getattr(self, name).update(bool((buttons >> bit) & 0x1))

        self.lx = self._axis(self.lx, struct.unpack_from("<f", data, 4)[0])
        self.rx = self._axis(self.rx, struct.unpack_from("<f", data, 8)[0])
        self.ry = self._axis(self.ry, struct.unpack_from("<f", data, 12)[0])
        self.l2 = self._axis(self.l2, struct.unpack_from("<f", data, 16)[0])
        self.ly = self._axis(self.ly, struct.unpack_from("<f", data, 20)[0])


@dataclass
class RobotState:
    qpos: np.ndarray
    qvel: np.ndarray
    tau: np.ndarray
    quat: np.ndarray
    rpy: np.ndarray
    gyro: np.ndarray
    mode_machine: int
    wireless_remote: bytes


class EmergencyStop(RuntimeError):
    pass


def _as_f32(values: Any, size: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if size is not None and arr.size != size:
        raise ValueError(f"expected {size} values, got {arr.size}")
    return arr.reshape(-1)


def rpy_to_projected_gravity(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy[:3]]

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array(
        [
            -2.0 * (x * z - y * w),
            -2.0 * (y * z + x * w),
            -1.0 + 2.0 * (x * x + y * y),
        ],
        dtype=np.float32,
    )


def quat_to_projected_gravity(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in np.asarray(quat, dtype=np.float32).reshape(-1)[:4]]
    return np.array(
        [
            -2.0 * (x * z - y * w),
            -2.0 * (y * z + x * w),
            -1.0 + 2.0 * (x * x + y * y),
        ],
        dtype=np.float32,
    )


def load_json_or_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return data


def load_pd_config(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_json_or_yaml(path)
    init_pos = _as_f32(data["init_pos"], G1_NUM_MOTOR)
    kp = _as_f32(data["kp"], G1_NUM_MOTOR)
    kd = _as_f32(data["kd"], G1_NUM_MOTOR)
    return init_pos, kp, kd


def load_loco_config(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_json_or_yaml(path)
    init_pos = _as_f32(data["init_pos"], G1_NUM_MOTOR)
    kp = _as_f32(data["kp"], G1_NUM_MOTOR)
    kd = _as_f32(data["kd"], G1_NUM_MOTOR)
    action_scale = _as_f32(data["action_scale"], G1_NUM_MOTOR)
    return init_pos, kp, kd, action_scale


def resolve_asset_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        (config_path.parent / path).resolve(),
        (Path.cwd() / path).resolve(),
        (config_path.parent / path.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve asset path {value!r} from {config_path}")


class OnnxRunner:
    def __init__(self, path: Path, providers: list[str] | None = None) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing onnxruntime. Install it before running policies.") from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        self.input_name = self.inputs[0].name if self.inputs else None
        self.expected_input_size = self._static_input_size()

    def _static_input_size(self) -> int | None:
        if not self.inputs:
            return None
        total = 1
        for dim in self.inputs[0].shape:
            if not isinstance(dim, int):
                continue
            total *= int(dim)
        return int(total) if total > 1 else None

    def run(self, obs: np.ndarray | None = None) -> np.ndarray:
        if self.input_name is None:
            result = self.session.run(None, {})
            return np.asarray(result[0], dtype=np.float32)

        if obs is None:
            raise ValueError("this ONNX model requires an observation input")
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.expected_input_size is not None and flat.size != self.expected_input_size:
            raise ValueError(
                f"ONNX input size mismatch: expected {self.expected_input_size}, got {flat.size}"
            )
        result = self.session.run(None, {self.input_name: flat.reshape(1, -1)})
        return np.asarray(result[0], dtype=np.float32).reshape(-1)
