# Unitree Go2 平地行走任务迁移 - 完整教学

## 目录
1. [项目背景](#项目背景)
2. [两个框架对比](#两个框架对比)
3. [迁移步骤详解](#迁移步骤详解)
4. [关键问题分析](#关键问题分析)
5. [代码详细解析](#代码详细解析)
6. [训练和验证](#训练和验证)

---

## 项目背景

### 什么是这个项目？
这个项目的目标是将 Unitree Go2 四足机器人的**平地行走任务**从一个强化学习框架（IsaacLab）迁移到另一个框架（Motrix）。

### 为什么要做迁移？
- **IsaacLab**: 基于 NVIDIA Isaac Sim 的强化学习框架，功能强大但依赖较重
- **Motrix**: 轻量级的强化学习环境框架，可能更适合特定的应用场景

### 任务目标
训练 Go2 机器人在平地上行走，使其能够：
- 跟踪线速度指令（x, y 方向）
- 跟踪角速度指令（yaw 旋转）
- 保持稳定的步态
- 避免摔倒或碰撞

---

## 两个框架对比

### 文件结构对比

| IsaacLab | Motrix | 说明 |
|----------|--------|------|
| `flat_env_cfg.py` + `rough_env_cfg.py` | `cfg.py` | 环境配置 |
| 继承式设计（flat 继承 rough） | 单文件配置 | 组织方式 |
| `walk_np.py` | `walk_np.py` | 环境实现类 |

### 配置结构对比

**IsaacLab 配置特点：**
```python
@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 通过覆写父类属性来定制
        self.rewards.flat_orientation_l2.weight = -2.5
        self.actions.joint_pos.scale = 0.25
```

**Motrix 配置特点：**
```python
@dataclass
class ControlConfig:
    action_scale = 0.25

@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "orientation": -2.5,
            # 其他奖励权重...
        }
    )
```

---

## 迁移步骤详解

### 第一步：理解 IsaacLab 原始配置

查看 `IsaacLab_go2/rough_env_cfg.py` 和 `flat_env_cfg.py`：

**关键参数识别：**
1. **动作尺度** (`action_scale`): 0.25
2. **奖励函数权重**：
   - 线速度跟踪: 1.5
   - 角速度跟踪: 0.75
   - 姿态惩罚: -2.5
   - 足部腾空时间: 0.25
   - 关节加速度: -2.5e-7
   - 力矩惩罚: -0.0002
3. **速度指令范围**：
   - x 方向: [-2.0, 2.0] m/s
   - y 方向: [-1.0, 1.0] m/s
   - 角速度: [-π, π] rad/s

### 第二步：创建 Motrix 配置

在 `cfg.py` 中对应设置参数：

```python
@dataclass
class ControlConfig:
    action_scale = 0.25  # ← 从 IsaacLab 迁移

@dataclass
class Commands:
    vel_limit = [
        [-1.0, -1.0, -1.0],  # ← 简化为对称范围
        [1.0, 1.0, 1.0],
    ]

@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 1.5,    # ← IsaacLab 的 1.5
            "tracking_ang_vel": 0.75,   # ← IsaacLab 的 0.75
            "orientation": -2.5,        # ← IsaacLab 的 -2.5
            "feet_air_time": 0.25,      # ← IsaacLab 的 0.25
            "dof_acc": -2.5e-7,         # ← IsaacLab 的 -2.5e-7
            "torques": -0.0,            # ← 有意设为 0（框架差异）
            "action_rate": -0.01,       # ← 10倍于原值（经验证有效）
            # ...
        }
    )
```

### 第三步：修复动作空间定义（最关键！）

**问题发现：**

在 `walk_np.py` 的 `_init_action_space` 方法中，原始代码使用了物理关节限位：

```python
# 错误的实现（修改前）
def _init_action_space(self):
    model = self.model
    self._action_space = gym.spaces.Box(
        np.array(model.actuator_ctrl_limits[0, :]),  # ← 使用物理限位
        np.array(model.actuator_ctrl_limits[1, :]),  # ← 这是错的！
        (model.num_actuators,),
        dtype=np.float32,
    )
```

**为什么这是错的？**

1. 打印 `actuator_ctrl_limits` 后发现：
   - calf 关节范围: [-2.62, -0.85]（全是负值！）
   - 其他关节范围也不是 [-1, 1]

2. 这导致的问题：
   - 动作空间不对称，小腿抬起空间极度受限
   - 与 IsaacLab 的语义不等价（IsaacLab 用的是 [-1, 1]）
   - 无法正确表达"归一化动作 × action_scale"的设计意图

3. 为什么之前训练能成功？
   - RL 策略自适应地学会了输出较小的值
   - 属于"歪打正着"，不是正确的解决方案

**正确的修复：**

```python
# 正确的实现（修改后）
def _init_action_space(self):
    model = self.model
    self._action_space = gym.spaces.Box(
        -np.ones(model.num_actuators),  # ← 固定为 -1
        np.ones(model.num_actuators),   # ← 固定为 +1
        (model.num_actuators,),
        dtype=np.float32,
    )
```

这样，动作空间就是标准的 `[-1, 1]`，与 IsaacLab 完全一致。

---

## 关键问题分析

### 问题 1: action_rate 参数变化

**观察到的现象：**
- 原版 Motrix: `-0.001`
- 迁移后: `-0.01`（放大 10 倍）

**原因分析：**
- 这是迁移过程中使用 AI 辅助时的无意改动
- 并非来自 IsaacLab 原始配置

**处理方式：**
- 经实际训练验证，该改动对结果无负面影响
- 决定保留该值

### 问题 2: torques 惩罚设为 0

**IsaacLab 中的值：** `-0.0002`
**Motrix 中的值：** `0.0`

**为什么不同？**
- 两个框架对力矩的计算方式不同
- 在 flat（平地）任务中，关闭该项对训练结果无显著影响
- 经过实践验证，设为 0 是合理的选择

### 问题 3: 速度指令范围调整

**IsaacLab 原始范围：**
- x: [-2.0, 2.0] m/s
- y: [-1.0, 1.0] m/s
- 角速度: [-π, π] rad/s

**Motrix 调整后：**
- 全部: [-1.0, 1.0]

**调整原因：**
- 统一为三轴对称的范围
- 训练分布更保守、更均匀
- 降低训练难度，更容易收敛

---

## 代码详细解析

### 配置文件 (cfg.py)

#### 1. 控制配置
```python
@dataclass
class ControlConfig:
    action_scale = 0.25  # 动作缩放系数
```

**工作原理：**
```
目标关节角度 = action_scale × 网络输出动作 + 默认关节角度
```

例如：
- 网络输出: `action = 0.5`
- 默认角度: `default = 0.9` (thigh 关节)
- 目标角度: `0.25 × 0.5 + 0.9 = 1.025` rad

#### 2. 奖励配置
```python
@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 1.5,     # 跟踪线速度的奖励
            "tracking_ang_vel": 0.75,    # 跟踪角速度的奖励
            "orientation": -2.5,         # 姿态偏离的惩罚
            "feet_air_time": 0.25,       # 足部腾空时间的奖励
            "dof_acc": -2.5e-7,          # 关节加速度的惩罚
            "action_rate": -0.01,        # 动作变化率的惩罚
            "torques": -0.0,             # 力矩惩罚（在 Motrix 中设为 0）
            # ...
        }
    )
```

**奖励函数解析：**
- **正权重**：鼓励该行为（如跟踪速度指令）
- **负权重**：惩罚该行为（如姿态偏离、过大的关节加速度）
- 权重的绝对值越大，该项在总奖励中的影响越大

#### 3. 初始状态配置
```python
@dataclass
class InitState:
    pos = [0.0, 0.0, 0.42]  # 机器人初始高度 0.42m

    default_joint_angles = {
        "FL_hip": 0.1,      # 前左髋关节
        "FL_thigh": 0.9,    # 前左大腿关节
        "FL_calf": -1.8,    # 前左小腿关节
        # ... 其他 9 个关节
    }
```

这些角度定义了机器人的"站立姿态"。

### 环境实现文件 (walk_np.py)

#### 关键修改：动作空间初始化

**修改前的问题代码：**
```python
def _init_action_space(self):
    model = self.model
    self._action_space = gym.spaces.Box(
        np.array(model.actuator_ctrl_limits[0, :]),  # 问题！
        np.array(model.actuator_ctrl_limits[1, :]),  # 问题！
        (model.num_actuators,),
        dtype=np.float32,
    )
```

**问题所在：**
- `actuator_ctrl_limits` 是**物理关节限位**，例如：
  - calf 关节: [-2.62, -0.85]（小腿只能向一个方向弯曲）
  - hip 关节: [-1.05, 1.05]
  - thigh 关节: [-0.66, 2.97]
- 这些范围不对称、不归一化，无法配合 `action_scale=0.25` 使用

**修改后的正确代码：**
```python
def _init_action_space(self):
    model = self.model
    self._action_space = gym.spaces.Box(
        -np.ones(model.num_actuators),  # 统一为 -1
        np.ones(model.num_actuators),   # 统一为 +1
        (model.num_actuators,),
        dtype=np.float32,
    )
```

**为什么这样是对的：**
1. 动作空间统一为 `[-1, 1]`（归一化）
2. 与 IsaacLab 的设计完全一致
3. 配合 `action_scale=0.25` 可以正确缩放
4. 网络输出的动作语义清晰：0 表示默认姿态，±1 表示最大偏移

#### 调试用的打印语句

```python
def __init__(self, ...):
    # ...
    self._init_dof_pos = self._model.compute_init_dof_pos()
    print("actuator_ctrl_limits:", self._model.actuator_ctrl_limits)  # 调试用
    self._init_buffer()
```

这行打印帮助我们发现了问题所在。

---

## 关键问题分析

### 核心问题：动作空间的设计哲学

#### IsaacLab 的设计
```
动作输出 ∈ [-1, 1] (归一化)
    ↓ (乘以 action_scale)
目标偏移 ∈ [-0.25, 0.25]
    ↓ (加上默认角度)
最终目标角度
    ↓ (PD 控制器)
实际力矩输出
```

#### 错误的 Motrix 实现（修改前）
```
动作输出 ∈ [actuator_ctrl_limits] (不归一化，不对称！)
    ↓ (乘以 action_scale = 0.25)
这根本不对！因为 action 已经不是 [-1, 1] 了
```

#### 正确的 Motrix 实现（修改后）
```
动作输出 ∈ [-1, 1] (归一化) ✓
    ↓ (乘以 action_scale = 0.25)
目标偏移 ∈ [-0.25, 0.25] ✓
    ↓ (加上默认角度)
最终目标角度 ✓
```

### 为什么错误实现仍能训练成功？

1. **RL 的鲁棒性**：强化学习算法会自适应环境
2. **策略自我调整**：网络学会了输出较小的值来避免越界
3. **侥幸成功**：虽然能用，但不是正确的设计
4. **隐患**：
   - 某些关节的动作空间受限（如 calf）
   - 训练效率可能降低
   - 迁移的语义不一致

---

## 代码详细解析

### 完整的参数迁移表

| 参数名 | IsaacLab 值 | Motrix 修改前 | Motrix 修改后 | 说明 |
|--------|-------------|---------------|---------------|------|
| action_scale | 0.25 | 0.05 | 0.25 | ✓ 对齐 |
| tracking_lin_vel | 1.5 | 1.0 | 1.5 | ✓ 对齐 |
| tracking_ang_vel | 0.75 | 0.5 | 0.75 | ✓ 对齐 |
| orientation | -2.5 | 0.0 | -2.5 | ✓ 对齐 |
| feet_air_time | 0.25 | 1.0 | 0.25 | ✓ 对齐 |
| dof_acc | -2.5e-7 | -2.5e-7 | -2.5e-7 | ✓ 已对齐 |
| torques | -0.0002 | -0.00001 | 0.0 | ⚠️ 有意不同 |
| action_rate | -0.001 | -0.001 | -0.01 | ⚠️ 有意调整 |

### 动作空间修复的技术细节

**修改的代码位置：** `walk_np.py:62-69`

**修改前：**
```python
def _init_action_space(self):
    model = self.model
    actuator_ctrl_limits = np.stack(
        [model.actuator_ctrl_limits[0, :], model.actuator_ctrl_limits[1, :]]
    )
    self._action_space = gym.spaces.Box(
        actuator_ctrl_limits[0],  # 形如 [-2.62, -1.05, -0.66, ...]
        actuator_ctrl_limits[1],  # 形如 [-0.85, 1.05, 2.97, ...]
        (model.num_actuators,),
        dtype=np.float32,
    )
```

**修改后：**
```python
def _init_action_space(self):
    model = self.model
    self._action_space = gym.spaces.Box(
        -np.ones(model.num_actuators),  # 统一 [-1, -1, -1, ...]
        np.ones(model.num_actuators),   # 统一 [1, 1, 1, ...]
        (model.num_actuators,),
        dtype=np.float32,
    )
```

**具体影响分析：**

Go2 机器人有 12 个执行器（4条腿 × 3个关节）：
- 4 个 hip (髋关节)
- 4 个 thigh (大腿关节)
- 4 个 calf (小腿关节)

如果使用物理限位作为动作空间：
```
FL_calf: [-2.62, -0.85]  → 动作只能在负值范围
FR_calf: [-2.62, -0.85]  → 严重限制了小腿的运动
RL_calf: [-2.62, -0.85]
RR_calf: [-2.62, -0.85]
```

修复后，所有关节的动作空间统一为 `[-1, 1]`，语义清晰：
- `-1`: 最大负向偏移（-0.25 rad）
- `0`: 保持默认姿态
- `+1`: 最大正向偏移（+0.25 rad）

---

## 训练和验证

### 训练流程

1. **环境注册**
```python
@registry.envcfg("go2-flat-terrain-walk")
class Go2WalkNpEnvCfg(EnvCfg):
    # 配置参数
```

2. **启动训练**（具体命令取决于 Motrix 框架）
```bash
# 示例命令（需要根据实际框架调整）
python train.py --env go2-flat-terrain-walk
```

3. **观察训练指标**
- episode reward: 应逐渐增加
- tracking_lin_vel reward: 应接近最大值
- orientation penalty: 应逐渐减小
- 机器人应能稳定行走

### 验证结果

**成功的标志：**
- ✓ 机器人能够跟踪线速度指令（x, y 方向）
- ✓ 机器人能够跟踪角速度指令（yaw 旋转）
- ✓ 步态稳定，不会摔倒
- ✓ 姿态保持直立（orientation penalty 较小）

---

## 总结

### 迁移的核心要点

1. **参数对齐**：将 IsaacLab 的关键参数正确迁移到 Motrix
2. **动作空间修复**：这是唯一的实质性 bug，必须修复
3. **框架差异理解**：某些参数需要根据框架特性调整（如 torques）
4. **实践验证**：所有修改都需要通过训练验证

### 学习要点

1. **动作空间设计**：归一化动作空间 `[-1, 1]` 是标准做法
2. **配合 action_scale**：`0.25` 是合理的缩放系数，给网络足够的控制范围
3. **奖励函数设计**：平衡各项权重，引导策略学习期望行为
4. **框架迁移思维**：不能简单照搬，需要理解两个框架的异同

### 关键文件清单

- `修改前的/cfg.py`: 原始 Motrix 配置（有问题）
- `修改后的/cfg.py`: 修正后的 Motrix 配置
- `修改前的/walk_np.py`: 原始环境实现（动作空间有问题）
- `修改后的/walk_np.py`: 修正后的环境实现
- `IsaacLab_go2/flat_env_cfg.py`: IsaacLab 原始配置（参考）
- `IsaacLab_go2/rough_env_cfg.py`: IsaacLab 基础配置（参考）

---

## 下一步

如果您想要：
1. **运行训练**：需要配置 Motrix 环境和依赖
2. **进一步优化**：可以调整奖励函数权重
3. **扩展到 rough terrain**：参考 rough_env_cfg.py 进行类似迁移
4. **部署到实体机器人**：需要 sim-to-real 的额外工作

如有任何问题，欢迎随时询问！
