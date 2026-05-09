# G1 State Machine

本文档说明当前仓库中 `state_machine.py` 的运行流程，重点介绍：

- 状态机各模式的切换逻辑
- `loco` 和 `dance` 模型的输入输出处理
- `dance` 模型在部署时与 tracking `state_dict` 对齐的输入构成

## 状态机模式与切换逻辑

当前状态流转是：

```text
DAMPING -> STAND -> WAIT_LOCO -> LOCO -> DANCE
                           ^         |       |
                           |         |       |
                           +---------+-------+
```

任意时刻：

- 遥控器 `select`
- 进程收到 `SIGINT` / `SIGTERM`

都会触发 `EmergencyStop`，并下发阻尼控制。

### 1. DAMPING

程序启动后先进入阻尼等待：

- 周期性调用 `_write_damping()`
- 等待第一帧 `LowState`
- 等待遥控器 `start`


### 2. STAND

按下 `start` 后，机器人从当前关节位置线性插值到默认站姿：

- 起点：当前 `qpos`
- 终点：`stand_qpos`
- 时长：`--stand-duration-s`，默认 3 秒


### 3. WAIT_LOCO

站起来以后，程序保持站姿输出：

- 持续下发 `stand_qpos`
- 等待遥控器 `A`
- 按下 `A` 进入 `LOCO`

### 4. LOCO

进入 `LOCO` 时先重置 loco 控制器状态：

- 清空 `last_action`
- 清空 `last_command`
- 记录当前双臂关节角，后续用于双臂 blend

运行中：

- 每个控制周期都调用 `self.loco.calculate(state, self.gamepad)`
- 输出目标关节角
- 用 loco 的 `kp/kd` 下发 PD 控制

按键逻辑：

- `X`：进入 `LOCO -> DANCE` 过渡
- `select`：急停

### 5. LOCO -> DANCE 过渡

按下 `X` 后，不会立刻切到 dance 推理，而是先做一个双臂过渡：

- 腿部和腰部仍然由 `loco` 输出控制
- 双臂从当前姿态线性插值到 dance 参考轨迹第 0 帧的双臂姿态
- 过渡时间：`--loco-to-dance-arm-blend-s`，默认 2 秒

过渡完成后：

- 调用 `enter_dance()`
- 用当前 `qpos` 初始化 dance 的 `last_motor_targets`
- `inference_counter` 清零

### 6. DANCE

进入 `DANCE` 后：

- 每个控制周期都调用 `self.dance.calculate(state)`
- 输出目标关节角
- 用 dance 配置里的 `kp/kd` 下发 PD 控制

退出条件有两个：

- 遥控器 `A`：立即返回 `LOCO`
- 参考轨迹跑完：`self.dance.done = True`，自动返回 `LOCO`

## DDS 输入状态是怎么来的

`state_machine.py` 的 DDS 回调 `_low_state_handler()` 会把机器人低层状态解析成 `RobotState`：

- `qpos`：29 维关节角
- `qvel`：29 维关节速度
- `quat`：4 维 IMU 四元数
- `rpy`：3 维欧拉角
- `gyro`：3 维 IMU 角速度
- `mode_machine`
- `wireless_remote`

## LOCO 模型的输入输出

### 输入

`LocoController.calculate()` 每拍构造一维 observation：

```text
obs =
[
  command(3),
  projected_gravity(3),
  gyro(3),
  qpos - default_qpos(29),
  qvel(29),
  last_action(29),
]
```

总维度：

- `3 + 3 + 3 + 29 + 29 + 29 = 96`

各字段含义：

- `command(3)`：
  - `ly * 0.8`
  - `lx * 0.4`
  - `rx * 0.8`
  之后还会做一步变化率限制
- `projected_gravity(3)`：由 `state.rpy` 转换得到
- `gyro(3)`：当前 IMU 角速度
- `qpos - default_qpos(29)`：当前关节偏离默认站姿的量
- `qvel(29)`：当前关节速度
- `last_action(29)`：上一轮 loco 模型输出

### 输出

`loco` 模型输出 29 维动作，当前代码会：

```text
target = default_qpos + action * action_scale
```

其中：

- `action_scale` 来自 `loco.json`
- 最终输出的是 29 个关节的目标角度
- 不是直接输出力矩

### 双臂进入过渡

进入 `LOCO` 后的前一段时间，双臂会从进入时的当前姿态渐变到 `loco` 默认双臂姿态：

- 时长：`--loco-arm-blend-s`
- 只影响双臂关节

## DANCE 模型的输入输出

### 输入：162 维

`DanceController` 按 `track_mj/envs/g1_tracking/play/play_g1_env_tracking_general.py:327` 的完整 `state_dict` 字段组装 actor `state`，字段顺序按名称字母排序。

输入顺序是：

```text
obs =
[
  dif_joint_pos(29),
  dif_joint_vel_scaled(29),
  gvec_pelvis(3),
  gyro_pelvis_scaled(3),
  joint_pos_minus_default(29),
  joint_vel_scaled(29),
  last_motor_targets(29),
  ref_feet_height(4),
  ref_root_angvel_scaled(3),
  ref_root_height(1),
  ref_root_linvel_scaled(3),
]
```

总维度：

- `29 + 29 + 3 + 3 + 29 + 29 + 29 + 4 + 3 + 1 + 3 = 162`

各字段语义如下。

#### 1. `dif_joint_pos(29)`

当前参考轨迹帧关节角减当前机器人关节角。

#### 2. `dif_joint_vel_scaled(29)`

当前参考轨迹帧关节速度减当前机器人关节速度，再乘 `dif_joint_vel` scale。当前配置为 `0.05`。

#### 3. `gvec_pelvis(3)`

当前机器人 pelvis IMU 姿态对应的 projected gravity。部署端用 DDS IMU 四元数计算，与 play 环境的 `site_xmat.T @ [0, 0, -1]` 同一语义。

#### 4. `gyro_pelvis_scaled(3)`

当前 IMU 角速度，乘 `joint_vel` scale。当前配置为 `0.05`。

#### 5. `joint_pos_minus_default(29)`

当前关节角相对 `DEFAULT_QPOS` 的偏移。

#### 6. `joint_vel_scaled(29)`

当前关节速度乘 `joint_vel` scale。

#### 7. `last_motor_targets(29)`

上一拍实际下发给 dance 的目标关节角。进入 dance 时，它会先用当前 `qpos` 初始化。

#### 8. `ref_feet_height(4)`

当前参考轨迹帧四个足部 site 高度，顺序为 `left_foot/right_foot/left_foot_top/right_foot_top`。

#### 9. `ref_root_angvel_scaled(3)`

参考根部角速度乘 `joint_vel` scale。

#### 10. `ref_root_height(1)`

当前参考轨迹帧根部高度，即 `traj_data.qpos[2]`。

#### 11. `ref_root_linvel_scaled(3)`

参考根部线速度转到参考根坐标系后乘 `joint_vel` scale。

### 输出：29 维动作

`dance` 模型输出必须正好是 29 维。  

输出动作的语义不是绝对关节角，而是：

- **相对当前参考轨迹帧关节角的增量修正**

当前部署处理是：

```text
target[joint_id] = ref_qpos[joint_id] + action[i] * 1.0
```

也就是：

- 基准：参考轨迹当前帧 `ref_qpos`
- 修正量：模型输出 `action`
- 缩放：当前是 `1.0`


## 遥控器按键含义

当前状态机里直接参与模式切换的按键是：

- `start`：`DAMPING -> STAND`
- `A`：
  - `WAIT_LOCO -> LOCO`
  - `DANCE -> LOCO`
- `X`：`LOCO -> DANCE`
- `select`：任意时刻急停

摇杆只用于 `loco`：

- `ly`：前后速度命令
- `lx`：侧向速度命令
- `rx`：偏航速度命令



## 安全说明

- `stand.json` 的 `init_pos` 必须和 `loco.json` 的 `init_pos` 完全一致，否则程序会直接报错退出。
- 首次运行前，确认机器人处于安全支撑或悬挂状态。
- 任意时刻可以按 `select` 进入阻尼急停。



## 运行方式

```bash
python3 state_machine.py eno1
```
## 关键命令行参数
- `net`：DDS 网卡，默认 `eno1`
- `--dt`：控制周期，默认 `0.02`
- `--stand-duration-s`：站起插值时长，默认 `3.0`
- `--loco-arm-blend-s`：进入 loco 后双臂渐变时长，默认 `2.0`
- `--loco-to-dance-arm-blend-s`：loco 切 dance 前双臂过渡时长，默认 `2.0`
- `--providers`：ONNX Runtime providers
- `--no-release-motion`：跳过释放已有高层运动模式
- `--dance-reference`：dance 参考轨迹，默认 `storage/data/mocap/lafan1/UnitreeG1/dance1_subject3.npz`；必须是带 `qpos/qvel` 的 `.npz`

默认资源路径：
- `stand` 配置：`models/unitree_g1_fsm/stand/stand.json`
- `loco` 配置：`models/unitree_g1_fsm/loco/loco.json`
- `loco` 模型：`models/unitree_g1_fsm/loco/policy.onnx`
- `dance` 配置：`models/unitree_g1_fsm/dance/dance.json`
- `dance` 模型：`models/unitree_g1_fsm/dance/policy.onnx`
- `dance` 参考轨迹：`storage/data/mocap/lafan1/UnitreeG1/dance1_subject3.npz`
