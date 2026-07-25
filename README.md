# ACP 单电脑部署（Flexiv + 腕部相机）

本目录是一套独立部署实现，不依赖旧的 `acp_flexiv_deploy`。它在操作电脑的
RTX 5060 上运行两个本地进程：ACP 推理进程使用新建的推理环境，机器人进程
继续使用采集时的 Flexiv RDK / RealSense 环境。二者只通过
`tcp://127.0.0.1:5555` 通信。

固定实验对象如下：

- Flexiv：`Rizon4s-063586`，工具 `hapticexoteleop`；
- 自动初始关节角：`[0, -32, 0, 90, 0, 28, 45]` 度；
- 腕部相机：`260322274925`，数据名 `cam_260322274925_wrist`；
- 只使用 RGB，不使用深度、点云、主相机或夹爪；
- 不调用 `ZeroFTSensor`，ACP 始终接收未经基线扣除的 `ext_wrench_in_tcp`；
- `execute` 只运行一个 action chunk 的前 4 个稀疏点；连续模式每执行 2 点便重新
  采集观测和推理，但会保留当前 16 点计划，依次执行 `0-1、2-3...14-15`，避免
  每轮重新从动作前缀开始。

## 1. 把训练结果复制到操作电脑

从 5090 训练机复制完整的 `latest.ckpt`，建议放在操作电脑的独立目录，例如：

```bash
mkdir -p "$HOME/acp_checkpoints/force30_torque15"
# 使用你自己的 scp/移动硬盘命令，将 latest.ckpt 放到上面目录
sha256sum "$HOME/acp_checkpoints/force30_torque15/latest.ckpt"
```

组合启动器默认使用本机 2026-07-25 完成的 800-epoch 腕部训练 `latest.ckpt`，也可通过
`ACP_CHECKPOINT_PATH` 显式覆盖。默认配置只接受 checkpoint 内部名称标记为 `wrist`
且 epoch 不低于 700 的模型；2026-07-24 实验所用的 300-epoch 腕部模型会被拒绝。

## 2. 准备两个 Conda 环境

机器人环境沿用采集电脑上的环境，先确认其中仍能导入硬件 SDK：

```bash
conda run -n data_collect python -c "import flexivrdk, pyrealsense2; print(flexivrdk.__version__)"
conda run -n data_collect pip install -r ./requirements-robot.txt
```

输出的 Flexiv RDK 必须是 `1.9.x`。依赖文件不会安装或升级 `flexivrdk` 与
`pyrealsense2`。

推理环境建议以 RTX 5090 训练环境为基础重新创建。已验证的训练组合是 Python
3.10、PyTorch `2.7.1+cu128`、`zarr==2.18.3`、`numcodecs==0.13.1`；RTX 5060
上仍需先用小脚本确认当前驱动能实际运行对应 CUDA wheel，再安装 ACP 仓库依赖：

```bash
conda create -n acp_deploy python=3.10 -y
conda activate acp_deploy
# 按操作电脑驱动安装合适的 PyTorch CUDA wheel，不要直接安装 CPU 版 torch。
pip install -r ./requirements-inference.txt
pip install "zarr==2.18.3" "numcodecs==0.13.1"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

确保仓库中的 `adaptive_compliance_policy/` 是训练 checkpoint 对应的代码版本。
默认推理配置从仓库根目录解析该路径。

## 3. 启动前检查

1. 手动闭合夹爪并保持全程闭合，本程序不控制夹爪。
2. 确认腕部相机序列号：`rs-enumerate-devices -s` 应出现 `260322274925`。
3. 清空自动归位轨迹和机械臂周围工作空间。
4. 确认硬件急停可立即触及；任何异常运动首先按下紧急停止。
5. 机器人无 fault、工具 `hapticexoteleop` 已在 Flexiv 系统中配置。
6. 检查 `configs/robot.yaml` 中的速度、力、工作空间和刚度限制，首次不要放宽。

腕部相机采集 `640x480 BGR8`，读取后先转成 RGB，再按官方实机输入方式做
等比例缩放和中心裁剪，最终交给 checkpoint 的尺寸是 `224x224 RGB`。
观测以最新 RGB 时间戳为统一锚点：RGB 历史覆盖约 333 ms，pose 历史覆盖约
101 ms，wrench 历史覆盖约 163 ms。机器人状态在执行控制循环中持续写入 buffer；
如果任一目标时间附近没有样本，推理会停止而不会拼接跨 chunk 的断档历史。

训练数据的机器人 pose 中位周期约为 `10.07 ms`，所以 action stride 50 对应
`0.5035 s`。该周期直接写入推理配置，不再使用原 ACP runner 的 `0.002 s` 常量。
Diffusion 每次推理固定 seed 42，以减少相同观测下的随机轨迹分叉。

`test0724` 的日志还表明，右偏并不是 Flexiv 执行器凭空产生的：旧模型第一段预测的
reference y 已从 `-0.113 m` 变为 `-0.052 m`，12 点预测末端达到 `+0.016 m`。
因此修复部署时序后若新模型仍在 dry-run 输出同样的横向轨迹，应优先检查训练数据和模型，
不能通过增大 `execute_points` 解决。

每次首帧等待上限为 15 秒；第一次失败会关闭并重建 RealSense pipeline，最多尝试
两次，总启动上限为 35 秒。两次均失败时进入 FAULT，不会无限等待。

## 4. 必须先运行 dry-run

组合启动器顶部已经固定了环境名和 checkpoint 路径：

```bash
ACP_ENV="pyrite"
ROBOT_ENV="haptic_exo_env"
DEFAULT_CHECKPOINT_PATH="${HOME}/haptic_exo_teleop_ws/liuyang/Data/acp_checkpoints/2026.07.25_14.19.52_flip_up_new_conv_wrist_190hz_800ep/checkpoints/latest.ckpt"
CHECKPOINT_PATH="${ACP_CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"
```

环境名改变时修改 `run_single_pc.sh`；checkpoint 也可以通过
`ACP_CHECKPOINT_PATH` 临时指定。以下命令都从仓库根目录执行：

```bash
ACP_CHECKPOINT_PATH=/absolute/path/to/latest.ckpt bash ./run_single_pc.sh dry-run
```

`dry-run` 仍会真实连接机器人并自动归位，因此会先要求输入 `y`。之后它启动腕部
相机、记录 2 秒 wrench 基线、构建历史并完成一次真实 checkpoint 推理，但不会发送
模型产生的笛卡尔位姿。结束状态应为 `HOLD`，这是预期结果，不会自动恢复运行。

ACP 训练标签的刚度范围是 `200-5000 N/m`。配置要求策略刚度上限不超过内层平移
刚度，连接机器人时还会验证内层笛卡尔刚度不超过 Flexiv 报告的额定刚度。
原始等效目标必须位于启动点 `0.20 m` 内；经过单步限幅、真正可能发送的位姿还必须
位于启动点 `0.08 m` 内。这两项分别防止模型目标失控和实际命令超出保护范围。

至少检查：相机画面方向和 RGB 颜色正确、推理无超时、raw/delta wrench 未越界，
`events.jsonl` 中恰好有 4 条 `action_preview_point`、汇总事件的
`stiffness_clip_count` 为 0，并且没有频繁的位姿限幅。任何预览点发生
`stiffness_clipped` 都会使 dry-run 进入 FAULT。程序会在 dry-run 的 `frames/` 中
保存本次推理使用的腕部 RGB，不保存全帧视频，以免影响时序。

也可用两个终端分别诊断：

```bash
conda activate pyrite
bash ./run_inference.sh \
  "$HOME/haptic_exo_teleop_ws/liuyang/Data/acp_checkpoints/2026.07.25_14.19.52_flip_up_new_conv_wrist_190hz_800ep/checkpoints/latest.ckpt"
```

```bash
conda activate haptic_exo_env
bash ./run_dry_run.sh
```

## 5. 单段 execute

只有 dry-run 日志通过后再运行：

```bash
bash ./run_single_pc.sh execute
```

execute 有两道独立确认：第一次输入 `y` 才允许自动归位；首个有效动作生成后，必须
完整输入 `Rizon4s-063586` 才会发送策略位姿。程序只执行一个 chunk 的前 4 个点，
随后进入锁存 HOLD。若推理超时、相机/机器人状态异常、力矩越界或请求 ID 不一致，
程序不会使用旧动作代替，而是 HOLD 或 FAULT 并退出。

这里使用固定 Flexiv 内层刚度下的“等效 TCP 目标”近似 ACP 平移刚度。它不等价于
ACP 官方 ManipServer 的动态 6x6 刚度控制，日志中的 predicted/applied stiffness 和
equivalent/applied pose 必须分别理解。

## 6. 连续闭环执行

单段 `execute` 日志通过后，必须先运行连续 dry-run：

```bash
bash ./run_single_pc.sh continuous-dry-run
```

`continuous-dry-run` 会真实连接机器人并自动归位，但不会发送任何策略位姿。它每预览
2 个动作点便重新采集腕部图像、位姿与 wrench，并按严格递增的 request ID 调用推理。
新推理先作为候选计划记录；当前计划的 16 点会按 `0-1、2-3...14-15` 顺序执行完，
之后才切换到最新候选，防止碗底阶段反复执行下降前缀。运行上限是 120 秒，正常结束
原因应为 `runtime_limit_reached`。必须检查 `action_selected_point` 的 point 是否连续、
每轮保存的腕部 RGB、推理延迟、刚度、等效目标与工作空间预测；任何异常都不能进入
真实连续执行。

真实连续模式使用以下绝对 TCP 工作空间：`x=[0.55, 0.92] m`、
`y=[-0.14, 0.13] m`、`z=[0.04, 0.43] m`。每轮原始等效目标距离该轮起始 TCP
不得超过 `0.20 m`。所有原有力/力矩、传感器时效、刚度、速度和单步位姿限制继续
逐控制周期生效。连续运动最多运行 120 秒；`Ctrl+C` 是正常操作员停止，物理紧急停止
始终是最高优先级。

只有操作者已在现场、机器人周围和自动归位轨迹已清空、紧急停止可立即触及、夹爪已
手动闭合时，才能启动：

```bash
bash ./run_single_pc.sh continuous
```

输入 `y` 允许归位，收到首个有效动作后完整输入 `Rizon4s-063586` 才开始连续运动。
不得通过 SSH 自动启动真实 `continuous`；SSH 仅用于更新代码、查看日志和执行不发送
策略位姿的诊断。

## 7. 日志与恢复

每次机器人进程都会在仓库的 `logs/robot/` 下创建不覆盖的目录：

- `metadata.json`：模式、完整配置、返回码、停止原因、完成 chunk 数和已发送命令数；
- `events.jsonl`：状态转换、归位、基线、原始/差分 wrench、完整动作和命令；
- `timing.csv`：逐 chunk 的 request ID、推理延迟、动作周期、点数、命令数和累计时间；
- `frames/`：dry-run 和连续模式中实际送入推理的 `224x224 RGB` PNG。

HOLD 和 FAULT 都不会在进程内解除。排查日志和硬件状态后，重新启动两个进程并重新
通过确认。组合启动脚本只终止它自己创建的推理子进程，不会按名称批量杀死其他任务。

本仓库在 Windows 上只能完成纯 Python、协议和启动脚本验证；RTX 5060 checkpoint
加载、RealSense 取流、Flexiv 连接、自动归位和物理执行必须在操作电脑上逐级验收，
不能把离线测试通过当作实机部署已经成功。
