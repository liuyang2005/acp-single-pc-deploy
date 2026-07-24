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
- 第一次执行只运行一个 action chunk 的前 12 个稀疏点，然后锁存 HOLD。

## 1. 把训练结果复制到操作电脑

从 5090 训练机复制完整的 `latest.ckpt`，建议放在操作电脑的独立目录，例如：

```bash
mkdir -p "$HOME/acp_checkpoints/force30_torque15"
# 使用你自己的 scp/移动硬盘命令，将 latest.ckpt 放到上面目录
sha256sum "$HOME/acp_checkpoints/force30_torque15/latest.ckpt"
```

路径可以改变，但启动时必须显式传入 checkpoint，程序不会自动选择“最新”文件。

## 2. 准备两个 Conda 环境

机器人环境沿用采集电脑上的环境，先确认其中仍能导入硬件 SDK：

```bash
conda run -n data_collect python -c "import flexivrdk, pyrealsense2; print(flexivrdk.__version__)"
conda run -n data_collect pip install -r acp_single_pc_deploy/requirements-robot.txt
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
pip install -r acp_single_pc_deploy/requirements-inference.txt
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
每次首帧等待上限为 15 秒；第一次失败会关闭并重建 RealSense pipeline，最多尝试
两次，总启动上限为 35 秒。两次均失败时进入 FAULT，不会无限等待。

## 4. 必须先运行 dry-run

组合启动器顶部已经固定了环境名和 checkpoint 路径：

```bash
ACP_ENV="pyrite"
ROBOT_ENV="haptic_exo_env"
CHECKPOINT_PATH="${HOME}/haptic_exo_teleop_ws/liuyang/acp_checkpoints/latest.ckpt"
```

如环境名或 checkpoint 位置改变，只修改 `run_single_pc.sh` 顶部这三项。运行时只需：

```bash
bash acp_single_pc_deploy/run_single_pc.sh dry-run
```

`dry-run` 仍会真实连接机器人并自动归位，因此会先要求输入 `y`。之后它启动腕部
相机、记录 2 秒 wrench 基线、构建历史并完成一次真实 checkpoint 推理，但不会发送
模型产生的笛卡尔位姿。结束状态应为 `HOLD`，这是预期结果，不会自动恢复运行。

至少检查：相机画面方向和 RGB 颜色正确、推理无超时、raw/delta wrench 未越界、
预测位姿和刚度为有限值、没有 `stiffness_clipped` 或频繁的位姿限幅。程序会在
dry-run 的 `frames/` 中保存本次推理使用的腕部 RGB，不保存全帧视频，以免影响时序。

也可用两个终端分别诊断：

```bash
conda activate pyrite
bash acp_single_pc_deploy/run_inference.sh \
  "$HOME/haptic_exo_teleop_ws/liuyang/acp_checkpoints/latest.ckpt"
```

```bash
conda activate haptic_exo_env
bash acp_single_pc_deploy/run_dry_run.sh
```

## 5. 单段 execute

只有 dry-run 日志通过后再运行：

```bash
bash acp_single_pc_deploy/run_single_pc.sh execute
```

execute 有两道独立确认：第一次输入 `y` 才允许自动归位；首个有效动作生成后，必须
完整输入 `Rizon4s-063586` 才会发送策略位姿。程序只执行一个 chunk 的前 12 个点，
随后进入锁存 HOLD。若推理超时、相机/机器人状态异常、力矩越界或请求 ID 不一致，
程序不会使用旧动作代替，而是 HOLD 或 FAULT 并退出。

这里使用固定 Flexiv 内层刚度下的“等效 TCP 目标”近似 ACP 平移刚度。它不等价于
ACP 官方 ManipServer 的动态 6x6 刚度控制，日志中的 predicted/applied stiffness 和
equivalent/applied pose 必须分别理解。

## 6. 日志与恢复

每次机器人进程都会在 `acp_single_pc_deploy/logs/robot/` 下创建不覆盖的目录：

- `metadata.json`：模式、完整配置、返回码、停止原因和已发送命令数；
- `events.jsonl`：状态转换、归位、基线、原始/差分 wrench、完整动作和命令；
- `timing.csv`：推理/动作周期字段表，供后续扩展逐周期统计。
- `frames/`：dry-run 中实际送入本次推理的 `224x224 RGB` PNG。

HOLD 和 FAULT 都不会在进程内解除。排查日志和硬件状态后，重新启动两个进程并重新
通过确认。组合启动脚本只终止它自己创建的推理子进程，不会按名称批量杀死其他任务。

本仓库在 Windows 上只能完成纯 Python、协议和启动脚本验证；RTX 5060 checkpoint
加载、RealSense 取流、Flexiv 连接、自动归位和物理执行必须在操作电脑上逐级验收，
不能把离线测试通过当作实机部署已经成功。
