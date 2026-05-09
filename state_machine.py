#!/usr/bin/env python3
"""Unitree G1 DDS finite-state controller.

States:
  DAMPING -> wait start
  STAND   -> 3 s interpolation to default stand posture, then hold
  LOCO    -> locomotion policy; X switches to DANCE
  DANCE   -> dance policy; A switches back to LOCO; completion also returns LOCO

At any point, select/back triggers emergency damping and exits.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from g1_dance_controller import DanceController
from g1_loco_controller import LocoController
from common import (
    ARM_IDS,
    CONTROL_DT,
    G1_NUM_MOTOR,
    MODE_PR,
    EmergencyStop,
    Gamepad,
    RobotState,
    _as_f32,
    _field,
    load_pd_config,
)

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"


class G1StateMachine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.gamepad = Gamepad()
        self.stop_requested = False
        self.state_lock = threading.Lock()
        self.state: RobotState | None = None

        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient,
            )
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.utils.crc import CRC
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing unitree_sdk2py. Install Unitree SDK2 Python and CycloneDDS "
                "on the robot/control host before running this script."
            ) from exc

        self.MotionSwitcherClient = MotionSwitcherClient
        self.ChannelFactoryInitialize = ChannelFactoryInitialize
        self.ChannelPublisher = ChannelPublisher
        self.ChannelSubscriber = ChannelSubscriber
        self.LowCmdDefault = unitree_hg_msg_dds__LowCmd_
        self.LowCmdType = LowCmd_
        self.LowStateType = LowState_
        self.crc = CRC()
        self.publisher = None
        self.subscriber = None
        self.low_cmd = self.LowCmdDefault()

        providers = args.providers.split(",") if args.providers else None
        self.stand_qpos, self.stand_kp, self.stand_kd = load_pd_config(args.stand_config)
        self.loco = LocoController(
            args.loco_policy,
            args.loco_config,
            providers,
            arm_blend_s=args.loco_arm_blend_s,
            control_dt=args.dt,
        )
        if not np.array_equal(self.stand_qpos, self.loco.default_qpos):
            raise ValueError(
                f"stand init_pos in {args.stand_config} must exactly match "
                f"loco init_pos in {args.loco_config}"
            )
        self.dance = DanceController(
            args.dance_policy,
            args.dance_config,
            providers,
            reference_path=args.dance_reference,
        )

    def _release_motion_mode(self) -> None:
        try:
            switcher = self.MotionSwitcherClient()
            switcher.SetTimeout(5.0)
            switcher.Init()
            for _ in range(10):
                _status, result = switcher.CheckMode()
                name = result.get("name", "") if isinstance(result, dict) else ""
                if not name:
                    return
                print(f"Releasing active motion mode: {name}")
                switcher.ReleaseMode()
                time.sleep(1.0)
        except Exception as exc:
            print(f"Warning: failed to release motion mode: {exc}", file=sys.stderr)

    def _init_dds(self) -> None:
        if self.args.net:
            self.ChannelFactoryInitialize(0, self.args.net)
        else:
            self.ChannelFactoryInitialize(0)

        if not self.args.no_release_motion:
            self._release_motion_mode()

        self.publisher = self.ChannelPublisher(TOPIC_LOWCMD, self.LowCmdType)
        self.publisher.Init()

        self.subscriber = self.ChannelSubscriber(TOPIC_LOWSTATE, self.LowStateType)
        self.subscriber.Init(self._low_state_handler, 10)

    def _low_state_handler(self, msg: Any) -> None:
        motor_states = _field(msg, "motor_state")
        imu_state = _field(msg, "imu_state")

        qpos = np.array([float(_field(motor_states[i], "q")) for i in range(G1_NUM_MOTOR)], dtype=np.float32)
        qvel = np.array([float(_field(motor_states[i], "dq")) for i in range(G1_NUM_MOTOR)], dtype=np.float32)
        quat = _as_f32(_field(imu_state, "quaternion"), 4)
        rpy = _as_f32(_field(imu_state, "rpy"), 3)
        gyro = _as_f32(_field(imu_state, "gyroscope"), 3)
        remote = bytes(int(v) & 0xFF for v in _field(msg, "wireless_remote"))

        with self.state_lock:
            self.state = RobotState(
                qpos=qpos,
                qvel=qvel,
                quat=quat,
                rpy=rpy,
                gyro=gyro,
                mode_machine=int(_field(msg, "mode_machine")),
                wireless_remote=remote,
            )

    def _get_state(self) -> RobotState | None:
        with self.state_lock:
            if self.state is None:
                return None
            return RobotState(
                qpos=self.state.qpos.copy(),
                qvel=self.state.qvel.copy(),
                quat=self.state.quat.copy(),
                rpy=self.state.rpy.copy(),
                gyro=self.state.gyro.copy(),
                mode_machine=self.state.mode_machine,
                wireless_remote=bytes(self.state.wireless_remote),
            )

    def _wait_for_state(self, timeout_s: float = 10.0) -> RobotState:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self._get_state()
            if state is not None:
                return state
            time.sleep(0.01)
        raise TimeoutError(f"no {TOPIC_LOWSTATE} received within {timeout_s:.1f}s")

    def _write_pd(
        self,
        q_target: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        mode_machine: int,
    ) -> None:
        if self.publisher is None:
            raise RuntimeError("DDS publisher has not been initialized")

        q_target = _as_f32(q_target, G1_NUM_MOTOR)
        kp = _as_f32(kp, G1_NUM_MOTOR)
        kd = _as_f32(kd, G1_NUM_MOTOR)

        self.low_cmd.mode_pr = MODE_PR
        self.low_cmd.mode_machine = int(mode_machine)
        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].q = float(q_target[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = float(kp[i])
            self.low_cmd.motor_cmd[i].kd = float(kd[i])
            self.low_cmd.motor_cmd[i].tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def _write_damping(self, mode_machine: int, damping_kd: float = 10.0) -> None:
        self._write_pd(
            q_target=np.zeros(G1_NUM_MOTOR, dtype=np.float32),
            kp=np.zeros(G1_NUM_MOTOR, dtype=np.float32),
            kd=np.full(G1_NUM_MOTOR, damping_kd, dtype=np.float32),
            mode_machine=mode_machine,
        )

    def _install_signal_handlers(self) -> None:
        def _handler(_signum: int, _frame: Any) -> None:
            self.stop_requested = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _sleep_until_next(self, tick_start: float) -> None:
        elapsed = time.monotonic() - tick_start
        if elapsed < self.args.dt:
            time.sleep(self.args.dt - elapsed)

    def _state_and_remote(self) -> RobotState | None:
        state = self._get_state()
        if state is None:
            return None
        self.gamepad.update(state.wireless_remote)
        if self.gamepad.select.pressed:
            raise EmergencyStop("select pressed")
        if self.stop_requested:
            raise EmergencyStop("signal received")
        return state

    def _damping_tick(self) -> None:
        state = self._state_and_remote()
        if state is not None:
            self._write_damping(state.mode_machine, self.args.damping_kd)

    def wait_for_start(self) -> None:
        print("Waiting for lowstate...")
        self._wait_for_state()
        print("DAMPING: press remote start to stand.")
        while True:
            tick = time.monotonic()
            self._damping_tick()
            if self.gamepad.start.on_press:
                return
            self._sleep_until_next(tick)

    def interpolate_to_stand(self) -> None:
        state = self._wait_for_state()
        start_qpos = state.qpos.copy()
        steps = max(1, int(round(self.args.stand_duration_s / self.args.dt)))
        print(f"STAND: interpolating to default posture in {self.args.stand_duration_s:.1f}s.")
        for step in range(1, steps + 1):
            tick = time.monotonic()
            state = self._state_and_remote()
            if state is None:
                self._sleep_until_next(tick)
                continue
            alpha = float(step) / float(steps)
            target = start_qpos * (1.0 - alpha) + self.stand_qpos * alpha
            self._write_pd(target, self.stand_kp, self.stand_kd, state.mode_machine)
            self._sleep_until_next(tick)

    def wait_for_loco(self) -> None:
        print("STAND: press remote A to enter loco.")
        while True:
            tick = time.monotonic()
            state = self._state_and_remote()
            if state is not None:
                self._write_pd(self.stand_qpos, self.stand_kp, self.stand_kd, state.mode_machine)
            if self.gamepad.A.on_press:
                self.enter_loco()
                return
            self._sleep_until_next(tick)

    def enter_loco(self) -> None:
        state = self._wait_for_state()
        self.loco.reset(state.qpos)
        print("LOCO: X -> dance, select -> emergency stop.")

    def enter_dance(self) -> None:
        state = self._wait_for_state()
        self.dance.reset(state.qpos)
        print("DANCE: A -> loco, select -> emergency stop.")

    def transition_loco_to_dance(self, start_state: RobotState) -> None:
        start_arm_qpos = start_state.qpos[ARM_IDS].copy()
        target_arm_qpos = self.dance.first_frame_arm_qpos()
        duration_s = max(self.args.dt, float(self.args.loco_to_dance_arm_blend_s))
        print(
            "LOCO->DANCE: interpolating arms to dance first reference frame "
            f"in {duration_s:.1f}s."
        )

        start_time = time.monotonic()
        while True:
            tick = time.monotonic()
            state = self._state_and_remote()
            alpha = min(1.0, (tick - start_time) / duration_s)

            if state is not None:
                target = self.loco.calculate(state, self.gamepad)
                target[ARM_IDS] = start_arm_qpos * (1.0 - alpha) + target_arm_qpos * alpha
                self._write_pd(target, self.loco.kp, self.loco.kd, state.mode_machine)

            if alpha >= 1.0:
                return
            self._sleep_until_next(tick)

    def run_loco(self) -> str:
        while True:
            tick = time.monotonic()
            state = self._state_and_remote()
            if state is None:
                self._sleep_until_next(tick)
                continue
            if self.gamepad.X.on_press:
                self.transition_loco_to_dance(state)
                self.enter_dance()
                return "dance"
            target = self.loco.calculate(state, self.gamepad)
            self._write_pd(target, self.loco.kp, self.loco.kd, state.mode_machine)
            self._sleep_until_next(tick)

    def run_dance(self) -> str:
        while True:
            tick = time.monotonic()
            state = self._state_and_remote()
            if state is None:
                self._sleep_until_next(tick)
                continue
            if self.gamepad.A.on_press:
                self.enter_loco()
                return "loco"
            target = self.dance.calculate(state)
            self._write_pd(target, self.dance.kp, self.dance.kd, state.mode_machine)
            if self.dance.done:
                self.enter_loco()
                return "loco"
            self._sleep_until_next(tick)

    def run(self) -> None:
        self._install_signal_handlers()
        self._init_dds()
        try:
            self.wait_for_start()
            self.interpolate_to_stand()
            self.wait_for_loco()

            state_name = "loco"
            while True:
                if state_name == "loco":
                    state_name = self.run_loco()
                elif state_name == "dance":
                    state_name = self.run_dance()
                else:
                    raise RuntimeError(f"unknown state {state_name}")
        except EmergencyStop as exc:
            print(f"Emergency stop: {exc}")
            state = self._get_state()
            if state is not None:
                self._write_damping(state.mode_machine, self.args.damping_kd)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_models = root / "models" / "unitree_g1_fsm"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("net", nargs="?", default="eno1", help="DDS network interface, for example eth0")
    parser.add_argument("--dt", type=float, default=CONTROL_DT, help="control period in seconds")
    parser.add_argument("--providers", default="CPUExecutionProvider", help="comma-separated ONNX Runtime providers")
    parser.add_argument("--no-release-motion", action="store_true", help="do not release existing high-level motion mode")
    parser.add_argument("--damping-kd", type=float, default=10.0)

    parser.add_argument("--stand-duration-s", type=float, default=3.0)
    parser.add_argument("--stand-config", type=Path, default=default_models / "stand" / "stand.json")

    parser.add_argument("--loco-arm-blend-s", type=float, default=2.0)
    parser.add_argument("--loco-to-dance-arm-blend-s", type=float, default=2.0)
    parser.add_argument("--loco-policy", type=Path, default=default_models / "loco" / "policy.onnx")
    parser.add_argument("--loco-config", type=Path, default=default_models / "loco" / "loco.json")

    parser.add_argument("--dance-policy", type=Path, default=default_models / "dance" / "policy.onnx")
    parser.add_argument(
        "--dance-reference",
        type=Path,
        default=root / "storage" / "data" / "mocap" / "lafan1" / "UnitreeG1" / "dance1_subject3.npz",
        help="dance reference trajectory (.npz mocap or legacy .onnx)",
    )
    parser.add_argument(
        "--dance-config",
        type=Path,
        default=default_models / "dance" / "dance.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = G1StateMachine(args)
    controller.run()


if __name__ == "__main__":
    main()
