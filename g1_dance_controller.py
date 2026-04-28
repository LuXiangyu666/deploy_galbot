#!/usr/bin/env python3
"""Dance policy controller for the G1 DDS state machine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common import (
    ARM_IDS,
    G1_NUM_MOTOR,
    OnnxRunner,
    RobotState,
    _as_f32,
    load_json_or_yaml,
    quat_to_projected_gravity,
    resolve_asset_path,
)


class DanceController:
    def __init__(
        self,
        policy_path: Path,
        config_path: Path,
        providers: list[str] | None,
    ) -> None:
        self.policy = OnnxRunner(policy_path, providers)
        self.cfg = load_json_or_yaml(config_path)
        self.kp = _as_f32(self.cfg["KPs"][0], G1_NUM_MOTOR)
        self.kd = _as_f32(self.cfg["KDs"][0], G1_NUM_MOTOR)
        self.default_qpos = _as_f32(self.cfg["DEFAULT_QPOS"], G1_NUM_MOTOR)
        self.obs_joint_ids = np.asarray(self.cfg["OBS_JOINT_IDS"], dtype=np.int32).reshape(-1)
        self.obs_joint_num = int(self.obs_joint_ids.size)
        self.joint_vel_scale = float(self.cfg["joint_vel_scale"])
        self.action_scale = 1.0
        self.onnx_input_size = self.obs_joint_num * 4 + 9

        qpos_path = resolve_asset_path(config_path, str(self.cfg["ref_qpos_path"]))
        self.ref_qpos_all = OnnxRunner(qpos_path, providers).run()
        self.end_iter = int(self.ref_qpos_all.shape[0])

        if self.policy.expected_input_size not in (None, self.onnx_input_size):
            raise ValueError(
                f"dance policy input must be {self.onnx_input_size}, "
                f"got {self.policy.expected_input_size}"
            )
        if self.ref_qpos_all.ndim != 2 or self.ref_qpos_all.shape[1] < G1_NUM_MOTOR + 4:
            raise ValueError("dance qpos reference must have at least 33 columns")

        self.last_motor_targets = self.default_qpos.copy()
        self.inference_counter = 0
        self.done = False

    def reset(self, current_qpos: np.ndarray) -> None:
        self.last_motor_targets = _as_f32(current_qpos, G1_NUM_MOTOR).copy()
        self.inference_counter = 0
        self.done = False

    def _reference_rows_at(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        ref_qpos_row = np.asarray(self.ref_qpos_all[row], dtype=np.float32).reshape(-1)
        ref_qpos = ref_qpos_row[-G1_NUM_MOTOR:]
        ref_quat = ref_qpos_row[-G1_NUM_MOTOR - 4 : -G1_NUM_MOTOR]
        return ref_qpos, ref_quat

    def _reference_rows(self) -> tuple[np.ndarray, np.ndarray]:
        return self._reference_rows_at(self.inference_counter)

    def first_frame_arm_qpos(self) -> np.ndarray:
        ref_qpos, _ref_quat = self._reference_rows_at(0)
        return ref_qpos[ARM_IDS].copy()

    def _make_obs(self, state: RobotState) -> tuple[np.ndarray, np.ndarray]:
        ref_qpos, ref_quat = self._reference_rows()
        state_sensor: list[np.ndarray] = []
        state_sensor.append(ref_qpos[self.obs_joint_ids].astype(np.float32))
        state_sensor.append(quat_to_projected_gravity(ref_quat))
        state_sensor.append(quat_to_projected_gravity(state.quat))
        state_sensor.append((state.gyro.astype(np.float32) * self.joint_vel_scale))
        state_sensor.append((state.qpos.astype(np.float32) - self.default_qpos)[self.obs_joint_ids])
        state_sensor.append((state.qvel.astype(np.float32) * self.joint_vel_scale)[self.obs_joint_ids])
        state_sensor.append(self.last_motor_targets[self.obs_joint_ids].astype(np.float32))

        obs = np.concatenate(state_sensor).astype(np.float32)
        np.clip(obs, -100.0, 100.0, out=obs)
        if obs.size != self.onnx_input_size:
            raise RuntimeError(f"dance obs size mismatch: {obs.size} != {self.onnx_input_size}")
        return obs, ref_qpos

    def calculate(self, state: RobotState) -> np.ndarray:
        if self.inference_counter >= self.end_iter:
            self.done = True
            return self.default_qpos.copy()

        obs, ref_qpos = self._make_obs(state)
        action = self.policy.run(obs).astype(np.float32)
        if action.size != self.obs_joint_num:
            raise RuntimeError(
                f"dance policy returned {action.size} actions, expected exactly {self.obs_joint_num}"
            )
        target = self.default_qpos.copy()
        for i, joint_id in enumerate(self.obs_joint_ids):
            target[joint_id] = ref_qpos[joint_id] + action[i] * self.action_scale
        self.last_motor_targets = target.copy()
        self.inference_counter += 1
        return target.astype(np.float32)
