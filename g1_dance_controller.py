#!/usr/bin/env python3
"""Dance policy controller for the G1 DDS state machine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common import (
    ARM_IDS,
    CONTROL_DT,
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
        reference_path: Path | None = None,
        control_dt: float = CONTROL_DT,
    ) -> None:
        self.config_path = config_path
        self.policy = OnnxRunner(policy_path, providers)
        self.cfg = load_json_or_yaml(config_path)
        self.kp = _as_f32(self.cfg["KPs"][0], G1_NUM_MOTOR)
        self.kd = _as_f32(self.cfg["KDs"][0], G1_NUM_MOTOR)
        self.default_qpos = _as_f32(self.cfg["DEFAULT_QPOS"], G1_NUM_MOTOR)
        self.obs_joint_ids = np.asarray(self.cfg["OBS_JOINT_IDS"], dtype=np.int32).reshape(-1)
        self.action_joint_ids = np.asarray(
            self.cfg.get("ACTION_JOINT_IDS", self.cfg["OBS_JOINT_IDS"]),
            dtype=np.int32,
        ).reshape(-1)
        self.obs_joint_num = int(self.obs_joint_ids.size)
        self.action_joint_num = int(self.action_joint_ids.size)
        self.joint_vel_scale = float(self.cfg["joint_vel_scale"])
        self.action_scale = float(self.cfg.get("action_scale", 1.0))
        self.control_dt = float(control_dt)

        obs_scales = self.cfg.get("obs_scales", {})
        self.obs_joint_vel_scale = float(obs_scales.get("joint_vel", self.joint_vel_scale))
        self.dif_joint_vel_scale = float(obs_scales.get("dif_joint_vel", self.obs_joint_vel_scale))

        self.state_dict_keys = sorted([
            "dif_joint_pos",
            "dif_joint_vel",
            "gyro_pelvis",
            "gvec_pelvis",
            "joint_pos",
            "joint_vel",
            "last_motor_targets",
            "ref_feet_height",
            "ref_root_angvel",
            "ref_root_height",
            "ref_root_linvel",
        ])
        self.state_dict_dims = {
            "dif_joint_pos": G1_NUM_MOTOR,
            "dif_joint_vel": G1_NUM_MOTOR,
            "gyro_pelvis": 3,
            "gvec_pelvis": 3,
            "joint_pos": self.obs_joint_num,
            "joint_vel": self.obs_joint_num,
            "last_motor_targets": G1_NUM_MOTOR,
            "ref_feet_height": 4,
            "ref_root_angvel": 3,
            "ref_root_height": 1,
            "ref_root_linvel": 3,
        }
        self.obs_keys = list(self.cfg.get("obs_keys", self.state_dict_keys))
        self.history_keys = list(
            self.cfg.get(
                "history_keys",
                [
                    "gyro_pelvis",
                    "gvec_pelvis",
                    "joint_pos",
                    "joint_vel",
                ],
            )
        )
        self.history_len = int(self.cfg.get("history_len", 0))
        unknown_obs_keys = [key for key in self.obs_keys if key not in self.state_dict_dims]
        unknown_history_keys = [key for key in self.history_keys if key not in self.state_dict_dims]
        if unknown_obs_keys or unknown_history_keys:
            raise ValueError(
                f"unknown dance observation keys: obs={unknown_obs_keys}, history={unknown_history_keys}"
            )
        if self.obs_keys != self.state_dict_keys:
            raise ValueError(
                "dance obs_keys must exactly match tracking state_dict keys sorted alphabetically: "
                f"{self.state_dict_keys}, got {self.obs_keys}"
            )
        self.history_input_size = sum(self.state_dict_dims[key] for key in self.history_keys)
        self.tracking_input_size = (
            sum(self.state_dict_dims[key] for key in self.obs_keys)
            + self.history_len * self.history_input_size
        )
        if self.tracking_input_size != 162:
            raise ValueError(
                f"dance tracking input must be exactly 162 dims, got {self.tracking_input_size}"
            )
        if self.action_joint_num != G1_NUM_MOTOR:
            raise ValueError(f"dance action joints must be exactly {G1_NUM_MOTOR}, got {self.action_joint_num}")
        if not np.array_equal(self.action_joint_ids, np.arange(G1_NUM_MOTOR, dtype=np.int32)):
            raise ValueError("dance ACTION_JOINT_IDS must match OpenTrack ACTION_JOINT_NAMES actuator order")
        if not np.isclose(self.action_scale, 1.0):
            raise ValueError(f"dance action_scale must match OpenTrack value 1.0, got {self.action_scale}")

        ref_source = reference_path
        if ref_source is None:
            ref_source = resolve_asset_path(config_path, str(self.cfg["ref_qpos_path"]))
        else:
            ref_source = resolve_asset_path(config_path, str(reference_path))

        if ref_source.suffix != ".npz":
            raise ValueError(
                "tracking-aligned dance observations require a .npz reference trajectory "
                "with qpos/qvel, matching PlayG1TrackingGeneralEnv"
            )
        (
            self.ref_root_pos_all,
            self.ref_root_quat_all,
            self.ref_root_qvel_all,
            self.ref_qpos_all,
            self.ref_joint_vel_all,
        ) = self._load_reference_from_npz(ref_source)
        if self.ref_qpos_all.ndim != 2 or self.ref_qpos_all.shape[1] != G1_NUM_MOTOR:
            raise ValueError(f"dance reference qpos must be [T, {G1_NUM_MOTOR}], got {self.ref_qpos_all.shape}")
        if self.ref_root_pos_all.shape != (self.ref_qpos_all.shape[0], 3):
            raise ValueError(f"dance reference root position shape mismatch: {self.ref_root_pos_all.shape}")
        if self.ref_root_quat_all.shape != (self.ref_qpos_all.shape[0], 4):
            raise ValueError(f"dance reference root quaternion shape mismatch: {self.ref_root_quat_all.shape}")
        if self.ref_root_qvel_all.shape != (self.ref_qpos_all.shape[0], 6):
            raise ValueError(f"dance reference root qvel shape mismatch: {self.ref_root_qvel_all.shape}")
        if self.ref_joint_vel_all.shape != (self.ref_qpos_all.shape[0], G1_NUM_MOTOR):
            raise ValueError(f"dance reference joint velocity shape mismatch: {self.ref_joint_vel_all.shape}")
        self.end_iter = int(self.ref_qpos_all.shape[0])
        self.ref_feet_height_all: np.ndarray | None = None

        self._validate_policy_input_size()
        self._ensure_tracking_reference_features()

        self.last_motor_targets = self.default_qpos.copy()
        self.ref_feet_height_for_obs = np.zeros(4, dtype=np.float32)
        self.history_buffer: np.ndarray | None = None
        self.inference_counter = 0
        self.done = False

    @staticmethod
    def _joint_name_to_index() -> dict[str, int]:
        return {
            "left_hip_pitch_joint": 0,
            "left_hip_roll_joint": 1,
            "left_hip_yaw_joint": 2,
            "left_knee_joint": 3,
            "left_ankle_pitch_joint": 4,
            "left_ankle_roll_joint": 5,
            "right_hip_pitch_joint": 6,
            "right_hip_roll_joint": 7,
            "right_hip_yaw_joint": 8,
            "right_knee_joint": 9,
            "right_ankle_pitch_joint": 10,
            "right_ankle_roll_joint": 11,
            "waist_yaw_joint": 12,
            "waist_roll_joint": 13,
            "waist_pitch_joint": 14,
            "left_shoulder_pitch_joint": 15,
            "left_shoulder_roll_joint": 16,
            "left_shoulder_yaw_joint": 17,
            "left_elbow_joint": 18,
            "left_wrist_roll_joint": 19,
            "left_wrist_pitch_joint": 20,
            "left_wrist_yaw_joint": 21,
            "right_shoulder_pitch_joint": 22,
            "right_shoulder_roll_joint": 23,
            "right_shoulder_yaw_joint": 24,
            "right_elbow_joint": 25,
            "right_wrist_roll_joint": 26,
            "right_wrist_pitch_joint": 27,
            "right_wrist_yaw_joint": 28,
        }

    def _load_reference_from_npz(
        self,
        path: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        motion = np.load(path, allow_pickle=True)
        qpos = np.asarray(motion["qpos"], dtype=np.float32)
        qvel = np.asarray(motion["qvel"], dtype=np.float32)
        joint_names = [str(name) for name in motion["joint_names"].reshape(-1)]

        if qpos.ndim != 2 or qpos.shape[1] < 7:
            raise ValueError(f"invalid mocap qpos shape in {path}: {qpos.shape}")
        if qvel.ndim != 2 or qvel.shape[0] != qpos.shape[0] or qvel.shape[1] != qpos.shape[1] - 1:
            raise ValueError(f"invalid mocap qvel shape in {path}: {qvel.shape} for qpos {qpos.shape}")
        if not joint_names or joint_names[0] != "root":
            raise ValueError(f"expected joint_names[0] to be 'root' in {path}")

        if bool(self.cfg.get("recalculate_vel_in_reference_motion", True)):
            qvel = self._recalculate_reference_qvel(qpos, qvel)

        source_frequency = float(np.asarray(motion["frequency"]).item()) if "frequency" in motion else 1.0 / self.control_dt
        if source_frequency <= 0.0:
            raise ValueError(f"invalid mocap frequency in {path}: {source_frequency}")
        target_frequency = 1.0 / self.control_dt
        if not np.isclose(source_frequency, target_frequency):
            qpos, qvel = self._interpolate_reference_motion(qpos, qvel, source_frequency, target_frequency)

        ref_qpos = np.zeros((qpos.shape[0], G1_NUM_MOTOR), dtype=np.float32)
        ref_joint_vel = np.zeros((qpos.shape[0], G1_NUM_MOTOR), dtype=np.float32)
        joint_to_index = self._joint_name_to_index()

        expected_joint_count = qpos.shape[1] - 7
        mocap_joint_names = joint_names[1:]
        if len(mocap_joint_names) != expected_joint_count:
            raise ValueError(
                f"mocap joint_names mismatch in {path}: "
                f"{len(mocap_joint_names)} names for {expected_joint_count} joint columns"
            )

        for mocap_idx, joint_name in enumerate(mocap_joint_names):
            target_index = joint_to_index.get(joint_name)
            if target_index is None:
                continue
            ref_qpos[:, target_index] = qpos[:, 7 + mocap_idx]
            ref_joint_vel[:, target_index] = qvel[:, 6 + mocap_idx]

        ref_root_pos = qpos[:, 0:3].astype(np.float32, copy=False)
        ref_quat = qpos[:, 3:7].astype(np.float32, copy=False)
        ref_root_qvel = qvel[:, 0:6].astype(np.float32, copy=False)
        return (
            ref_root_pos,
            ref_quat,
            ref_root_qvel,
            ref_qpos.astype(np.float32, copy=False),
            ref_joint_vel.astype(np.float32, copy=False),
        )

    def _validate_policy_input_size(self) -> None:
        expected = self.policy.expected_input_size
        if expected not in (None, self.tracking_input_size):
            raise ValueError(
                f"dance policy input must match tracking state_dict obs_keys: "
                f"expected {self.tracking_input_size}, got {expected}"
            )

    def _recalculate_reference_qvel(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        qvel = qvel.copy()
        if qpos.shape[0] <= 1:
            return qvel

        frequency = 1.0 / self.control_dt
        qvel[1:, :3] = (qpos[1:, :3] - qpos[:-1, :3]) * frequency
        qvel[1:, 3:6] = self._quat_delta_angvel(qpos[:-1, 3:7], qpos[1:, 3:7]) * frequency
        qvel[1:, 6:] = (qpos[1:, 7:] - qpos[:-1, 7:]) * frequency
        return qvel.astype(np.float32, copy=False)

    @staticmethod
    def _interpolate_reference_motion(
        qpos: np.ndarray,
        qvel: np.ndarray,
        old_frequency: float,
        new_frequency: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            from scipy.interpolate import interp1d
            from scipy.spatial.transform import Rotation, Slerp
        except ModuleNotFoundError as exc:
            raise RuntimeError("reference interpolation requires scipy") from exc

        if qpos.shape[0] <= 1:
            return qpos, qvel

        sampling_factor = float(new_frequency) / float(old_frequency)
        old_t = np.arange(qpos.shape[0], dtype=np.float32)
        new_t = np.linspace(0.0, float(qpos.shape[0] - 1), round(qpos.shape[0] * sampling_factor), endpoint=True)

        qpos_interp = np.empty((new_t.shape[0], qpos.shape[1]), dtype=np.float32)
        non_quat_ids = [idx for idx in range(qpos.shape[1]) if idx < 3 or idx >= 7]
        qpos_interp[:, non_quat_ids] = interp1d(old_t, qpos[:, non_quat_ids], kind="cubic", axis=0)(new_t)

        # Match OpenTrack's interpolate_trajectories implementation exactly.
        qpos_interp[:, 3:7] = Slerp(old_t, Rotation.from_quat(qpos[:, 3:7]))(new_t).as_quat()

        qvel_interp = interp1d(old_t, qvel, kind="cubic", axis=0)(new_t).astype(np.float32)
        return qpos_interp.astype(np.float32, copy=False), qvel_interp

    @staticmethod
    def _quat_delta_angvel(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        q0_inv = np.concatenate([q0[:, :1], -q0[:, 1:]], axis=1)
        w1, x1, y1, z1 = q0_inv[:, 0], q0_inv[:, 1], q0_inv[:, 2], q0_inv[:, 3]
        w2, x2, y2, z2 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        angle = np.arccos(np.clip(2.0 * (w**2) - 1.0, -1.0, 1.0))
        axis = np.stack([x, y, z], axis=1)
        axis /= np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-9, None)
        return (axis * angle[:, None]).astype(np.float32, copy=False)

    @staticmethod
    def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
        q = np.outer(quat, quat)
        return np.array(
            [
                [
                    q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
                    2 * (q[1, 2] - q[0, 3]),
                    2 * (q[1, 3] + q[0, 2]),
                ],
                [
                    2 * (q[1, 2] + q[0, 3]),
                    q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
                    2 * (q[2, 3] - q[0, 1]),
                ],
                [
                    2 * (q[1, 3] - q[0, 2]),
                    2 * (q[2, 3] + q[0, 1]),
                    q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
                ],
            ],
            dtype=np.float32,
        )

    def _resolve_reference_mjcf_path(self) -> Path | None:
        configured = self.cfg.get("reference_mjcf_path")
        if configured:
            try:
                return resolve_asset_path(self.config_path, str(configured))
            except FileNotFoundError:
                pass

        candidates = [
            Path(__file__).resolve().parent / "storage" / "assets" / "unitree_g1" / "scene_mjx_flat_terrain.xml",
            Path(__file__).resolve().parent.parent / "OpenTrack" / "storage" / "assets" / "unitree_g1" / "scene_mjx_flat_terrain.xml",
            Path.cwd().parent / "OpenTrack" / "storage" / "assets" / "unitree_g1" / "scene_mjx_flat_terrain.xml",
        ]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists():
                return candidate
        return None

    def _compute_ref_feet_height_with_mujoco(self) -> np.ndarray:
        try:
            import mujoco
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "tracking-aligned dance observations require mujoco to compute ref_feet_height"
            ) from exc

        mjcf_path = self._resolve_reference_mjcf_path()
        if mjcf_path is None:
            raise FileNotFoundError(
                "tracking-aligned dance observations require reference_mjcf_path or a sibling OpenTrack asset tree"
            )

        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)
        site_names = list(
            self.cfg.get(
                "reference_feet_site_names",
                ["left_foot", "right_foot", "left_foot_top", "right_foot_top"],
            )
        )
        site_ids = np.array([model.site(name).id for name in site_names], dtype=np.int32)
        feet_height = np.zeros((self.ref_qpos_all.shape[0], len(site_ids)), dtype=np.float32)

        for idx in range(self.ref_qpos_all.shape[0]):
            data.qpos[:] = np.concatenate(
                [
                    self.ref_root_pos_all[idx],
                    self.ref_root_quat_all[idx],
                    self.ref_qpos_all[idx],
                ]
            ).astype(np.float32, copy=False)
            data.qvel.fill(0.0)
            mujoco.mj_forward(model, data)
            feet_height[idx] = data.site_xpos[site_ids, 2].astype(np.float32)
        return feet_height

    def _ensure_tracking_reference_features(self) -> None:
        if self.ref_feet_height_all is not None:
            return
        self.ref_feet_height_all = self._compute_ref_feet_height_with_mujoco()

    def reset(self, current_qpos: np.ndarray) -> None:
        _as_f32(current_qpos, G1_NUM_MOTOR)
        # Match PlayG1TrackingGeneralEnv._reset_from_current_traj:
        # last_motor_targets comes from reference frame 0, then traj_info is
        # advanced once before the first policy observation/action.
        self.last_motor_targets = self.ref_qpos_all[0].astype(np.float32, copy=True)
        self.ref_feet_height_for_obs = np.zeros(4, dtype=np.float32)
        self.history_buffer = None
        self.inference_counter = 1
        self.done = False

    def _reference_rows_at(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        ref_qpos = np.asarray(self.ref_qpos_all[row], dtype=np.float32).reshape(-1)
        ref_quat = np.asarray(self.ref_root_quat_all[row], dtype=np.float32).reshape(-1)
        return ref_qpos, ref_quat

    def _reference_rows(self) -> tuple[np.ndarray, np.ndarray]:
        return self._reference_rows_at(self.inference_counter)

    def first_frame_arm_qpos(self) -> np.ndarray:
        ref_qpos, _ref_quat = self._reference_rows_at(0)
        return ref_qpos[ARM_IDS].copy()

    def _make_tracking_state_dict(self, state: RobotState) -> tuple[dict[str, np.ndarray], np.ndarray]:
        row = self.inference_counter
        ref_qpos = self.ref_qpos_all[row].astype(np.float32, copy=False)
        ref_joint_vel = self.ref_joint_vel_all[row].astype(np.float32, copy=False)
        ref_root_pos = self.ref_root_pos_all[row].astype(np.float32, copy=False)
        ref_root_qvel = self.ref_root_qvel_all[row].astype(np.float32, copy=False)
        traj_root_rot_mat = self._quat_to_mat(self.ref_root_quat_all[row])
        ref_feet_height = self.ref_feet_height_for_obs.astype(np.float32, copy=False)

        state_dict = {
            "gyro_pelvis": state.gyro.astype(np.float32) * self.obs_joint_vel_scale,
            "gvec_pelvis": quat_to_projected_gravity(state.quat),
            "joint_pos": (state.qpos.astype(np.float32) - self.default_qpos)[self.obs_joint_ids],
            "joint_vel": state.qvel.astype(np.float32)[self.obs_joint_ids] * self.obs_joint_vel_scale,
            "last_motor_targets": self.last_motor_targets.astype(np.float32),
            "dif_joint_pos": ref_qpos - state.qpos.astype(np.float32),
            "dif_joint_vel": (ref_joint_vel - state.qvel.astype(np.float32)) * self.dif_joint_vel_scale,
            "ref_feet_height": ref_feet_height,
            "ref_root_height": np.array([ref_root_pos[2]], dtype=np.float32),
            "ref_root_linvel": (traj_root_rot_mat.T @ ref_root_qvel[:3]) * self.obs_joint_vel_scale,
            "ref_root_angvel": ref_root_qvel[3:6] * self.obs_joint_vel_scale,
        }
        return state_dict, ref_qpos

    def _make_tracking_obs(self, state: RobotState) -> tuple[np.ndarray, np.ndarray]:
        state_dict, ref_qpos = self._make_tracking_state_dict(state)
        obs = np.concatenate([state_dict[key] for key in self.obs_keys]).astype(np.float32)

        if self.history_len > 0:
            history = np.concatenate([state_dict[key] for key in self.history_keys]).astype(np.float32)
            if self.history_buffer is None:
                self.history_buffer = np.repeat(history[None, :], self.history_len, axis=0)
            obs = np.concatenate([obs, self.history_buffer.reshape(-1)], axis=0).astype(np.float32)
            self.history_buffer = np.concatenate([self.history_buffer[1:], history[None, :]], axis=0)

        if obs.size != self.tracking_input_size:
            raise RuntimeError(f"dance tracking obs size mismatch: {obs.size} != {self.tracking_input_size}")
        return obs, ref_qpos

    def calculate(self, state: RobotState) -> np.ndarray:
        if self.inference_counter >= self.end_iter:
            self.done = True
            return self.last_motor_targets.copy()

        obs, ref_qpos = self._make_tracking_obs(state)
        action = self.policy.run(obs).astype(np.float32)
        if action.size != self.action_joint_num:
            raise RuntimeError(
                f"dance policy returned {action.size} actions, expected exactly {self.action_joint_num}"
            )
        target = self.default_qpos.copy()
        for i, joint_id in enumerate(self.action_joint_ids):
            target[joint_id] = ref_qpos[joint_id] + action[i] * self.action_scale
        self.last_motor_targets = target.copy()
        self.ref_feet_height_for_obs = self.ref_feet_height_all[self.inference_counter].astype(np.float32, copy=True)
        self.inference_counter += 1
        return target.astype(np.float32)
