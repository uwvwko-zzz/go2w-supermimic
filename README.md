# Go2W SuperMimic

基于 mjlab 和 MuJoCo Warp 的 Go2W 轮足机器人强化学习项目。通过仿真数据训练世界模型，再利用世界模型生成想象轨迹来优化速度控制策略，支持平地、楼梯、平台、斜坡和碎石地形。

## 演示视频

https://github.com/user-attachments/assets/46d40ebe-21ee-4124-b646-20ba3c0927c9

## 世界模型

世界模型是带残差预测的前馈多层感知机（MLP），默认使用三层 512 维隐藏层和 SiLU 激活函数。它根据当前状态和动作预测下一时刻的状态变化，以及机身平面位移和偏航角变化。输入和输出使用运行统计量进行归一化。

| 输入 | 维度 | 内容 |
| --- | ---: | --- |
| 机器人状态 | 38 | 高度 1、投影重力 3、机身线速度 3、角速度 3、腿关节位置 12、腿关节速度 12、轮速 4 |
| 当前动作 | 16 | 12 维腿部位置控制和 4 维轮速控制 |
| 上一动作 | 16 | 上一控制周期的动作 |
| 物理参数 | 7 | 四个轮子的摩擦系数和机身质心偏移 |
| 地形扫描 | 273 | 复杂地形模型使用的 21 × 13 高度采样网格 |

平地世界模型输入为 **77 维**；复杂地形世界模型加入高度扫描后为 **350 维**。两者均输出 **41 维**：38 维状态变化和 3 维平面位姿变化。地形模型还支持可选的接触与接触力预测头。

训练时从仿真回放中采样连续片段，世界模型按时间自回归预测并拟合实际轨迹；默认训练窗口为 8 步。随后使用世界模型展开想象轨迹，并结合奖励和 critic 更新策略。

## 策略观测与历史

策略每帧接收 **53 维**观测：机身角速度 3、投影重力 3、腿关节位置 12、腿关节速度 12、轮速 4、上一动作 16 和速度指令 3。默认保留 **5 帧**历史，因此策略输入为 **265 维**；各观测项分别按时间从旧到新排列。控制周期为 0.02 秒，5 帧覆盖 5 个控制时刻。

策略网络默认隐藏层为 256、128、64 维，输出 16 维动作。策略不接收地形扫描；地形扫描及其他特权信息供训练时的世界模型和 critic 使用，实现盲走控制。策略历史与世界模型的 8 步训练窗口是两个不同的参数，分别由 `--history` 和 `--wm-window` 设置。

## 训练

在已配置 mjlab、PyTorch 和 CUDA 的 Python 环境中，从项目根目录启动。例如，复杂地形 v4 训练：

```bash
export PYTHONPATH="$PWD"
OMP_NUM_THREADS=1 MUJOCO_GL=egl python scripts/train.py \
  --terrain --terrain-revision 4 --num-envs 4096 --iterations 40000 \
  --num-steps-per-env 1 --buffer-len 512 \
  --wm-batch 2048 --wm-window 8 --wm-updates 4 \
  --imag-batch 8192 --horizon 32 \
  --critic-epochs 5 --critic-minibatches 4 \
  --compile-imagination --prioritized-replay --command-curriculum \
  --save-interval 500 --log-interval 10 --output logs/terrain_v4
```

该配置需要较大显存；可按硬件调整并行环境和批量大小。恢复训练使用 `--resume <model_N.pt>`，保留原地形版本和调度配置；`--iterations` 是目标总轮数。其他参数见 `python scripts/train.py --help`。

## 交互演示

```bash
export PYTHONPATH="$PWD"
./scripts/play_gui.sh --checkpoint logs/go2w_terrain_v4_40000/model_40000.pt \
  --terrain-kind flat --device cuda:0
```

在浏览器打开终端输出的地址。默认从 `http://127.0.0.1:8765/` 开始，端口占用时自动尝试下一个。方向键控制移动，Q/E 转向，空格停止；鼠标控制视角，界面中可切换地形与难度。按 `Ctrl+C` 退出。

## 代码结构

```text
go2w_supermimic/
├── core/    # 世界模型、策略、想象训练、经验回放与演示界面
├── env/     # 平地及复杂地形环境
└── robot/   # Go2W 机器人配置与模型资产
scripts/     # 训练与演示入口
assert/      # 演示视频
```

项目目前在仿真中验证。算法基于 [Go2-SuperMimic](https://github.com/Zhengtianjin/go2-supermimic)，仿真使用 [mjlab](https://github.com/mujocolab/mjlab)，Go2W 资产来自 `unitree_rl_mjlab`。
