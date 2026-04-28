#!/usr/bin/env python3
"""Locomotion policy controller for the G1 DDS state machine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common import (
    ARM_IDS,
    CONTROL_DT,
    G1_NUM_MOTOR,
    Gamepad,
    OnnxRunner,
    RobotState,
    _as_f32,
    load_loco_config,
    rpy_to_projected_gravity,
)


class LocoController:
    def __init__(
        self,
        policy_path: Path,
        config_path: Path,
        providers: list[str] | None,
        arm_blend_s: float = 2.0,
        control_dt: float = CONTROL_DT,
    ) -> None:
        self.policy = OnnxRunner(policy_path, providers)
        self.default_qpos, self.kp, self.kd, self.action_scale = load_loco_config(config_path)
        self.control_dt = float(control_dt)
        self.arm_blend_steps = max(1, int(round(arm_blend_s / self.control_dt)))
        self.last_action = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.last_command = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.entry_arm_qpos = self.default_qpos[ARM_IDS].copy()

    def reset(self, current_qpos: np.ndarray) -> None:
        self.last_action.fill(0.0)
        self.last_command.fill(0.0)
        self.step_count = 0
        self.entry_arm_qpos = _as_f32(current_qpos, G1_NUM_MOTOR)[ARM_IDS].copy()

    def calculate(self, state: RobotState, gamepad: Gamepad) -> np.ndarray:
        obs_command = np.array(
            [
                gamepad.ly * 0.8,
                gamepad.lx * 0.4,
                gamepad.rx * 0.8,
            ],
            dtype=np.float32,
        )

        delta = np.clip(obs_command - self.last_command, -self.control_dt, self.control_dt)
        obs_command = self.last_command + delta
        self.last_command = obs_command
        obs_cmd = np.array([obs_command[0], -obs_command[1], -obs_command[2]], dtype=np.float32)

        gvec_pelvis = rpy_to_projected_gravity(state.rpy)
        obs = np.concatenate(
            [
                obs_cmd,
                gvec_pelvis,
                state.gyro.astype(np.float32),
                state.qpos.astype(np.float32) - self.default_qpos,
                state.qvel.astype(np.float32),
                self.last_action,
            ]
        ).astype(np.float32)

        action = self.policy.run(obs)
        if action.size < G1_NUM_MOTOR:
            raise RuntimeError(f"loco policy returned {action.size} actions, expected 29")
        action = action[:G1_NUM_MOTOR].astype(np.float32)
        target = self.default_qpos + action * self.action_scale
        self.last_action = action.copy()

        if self.step_count < self.arm_blend_steps:
            alpha = float(self.step_count + 1) / float(self.arm_blend_steps)
            target[ARM_IDS] = self.entry_arm_qpos * (1.0 - alpha) + self.default_qpos[ARM_IDS] * alpha

        self.step_count += 1
        return target.astype(np.float32)
