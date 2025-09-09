 # USV 路径规划：基于强化学习与模仿学习

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green)

本项目是一个专注于无人水面艇（USV）路径规划的算法实现与实验平台。它利用先进的强化学习（RL）和模仿学习（IL）技术，旨在使 USV 能够自主、安全且高效地在包含静态或动态障碍物的复杂水域中导航。

## 🌟 功能特性

- **先进的算法实现**: 
  - **强化学习 (RL)**: 实现了经典的 **PPO (Proximal Policy Optimization)** 算法，通过与环境的交互和试错来学习最优导航策略。
  - **模仿学习 (IL)**: 包含 **GAIL (Generative Adversarial Imitation Learning)** 的相关模型检查点，可以从专家轨迹中学习导航行为。
- **可定制的仿真环境**: 
  - `env.py` 文件定义了一个灵活的 USV 仿真环境，支持自定义地图、障碍物、USV 初始状态等。
- **丰富的可视化工具**: 
  - `plot.py` 提供了强大的可视化功能，可以轻松绘制训练过程中的奖励曲线、成功率、航行轨迹、CPA (最近会遇距离) 等关键指标。
- **预训练模型**: 
  - 项目提供了多个在不同环境下（如 `env1`, `env4`）训练好的 PPO 和 GAIL 模型检查点（`.ckpt` 文件），可直接用于测试和评估。

## 📂 项目结构

```
gail_code/
├── env.py                  # 定义了强化学习环境（USV、障碍物、状态、动作、奖励等）
├── ppo.py                  # 实现了 PPO (Proximal Policy Optimization) 算法的训练与测试逻辑
├── plot.py                 # 用于绘制训练结果的脚本（奖励曲线、轨迹、CPA等）
├── env.pyc                 # Python 编译文件
├── *.ckpt                  # 预训练模型的检查点文件 (Checkpoint)
├── *.ipynb                 # 原始的 Jupyter Notebook 实验文件，用于快速验证和调试
└── __pycache__/            # Python 缓存目录
```

## ⚙️ 安装与环境配置

1. **克隆仓库**
   ```bash
   git clone <your-repo-url>
   cd gail_code
   ```

2. **创建虚拟环境 (推荐)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **安装依赖**
   项目主要依赖 PyTorch, NumPy, Pandas 和 Matplotlib。您可以通过 pip 安装它们：
   ```bash
   pip install torch numpy pandas matplotlib
   ```

## 🚀 使用指南

### 1. 训练新模型

您可以使用 `ppo.py` 脚本来从头开始训练一个新的 PPO 模型。

1. **配置训练参数**: 打开 `ppo.py` 文件，找到 `if __name__ == "__main__":` 代码块。
2. **修改参数**: 在此代码块中，您可以设置环境名称 (`env_name`)、学习率、训练回合数等超参数。确保 `restore` 设置为 `False`。
3. **开始训练**: 在终端中运行以下命令：
   ```bash
   python ppo.py
   ```
   训练日志将保存在 `ppo_<env_name>_log.csv` 中，模型检查点将根据 `save_model_freq` 的频率保存在 `.ckpt` 文件中。

### 2. 测试预训练模型

如果您想使用项目提供的预训练模型进行测试或演示：

1. **配置测试参数**: 打开 `ppo.py` 文件，找到 `if __name__ == "__main__":` 代码块。
2. **启用恢复模式**: 
   - 将 `restore` 变量设置为 `True`。
   - 将 `checkpoint_path` 变量设置为您想要加载的模型文件路径，例如 `'env4_ppo.ckpt'`。
   - 您可以将 `render` 变量设置为 `True` 来实时查看 USV 的导航过程（需要环境支持图形化渲染）。
3. **运行测试**:
   ```bash
   python ppo.py
   ```
   脚本将加载指定的 `.ckpt` 文件，并运行测试。

### 3. 可视化训练结果

使用 `plot.py` 脚本可以方便地将训练数据转换成图表。

1. **配置绘图参数**: 打开 `plot.py` 文件，找到 `main()` 函数。
2. **指定文件**: 修改 `algo_name` 和 `env_name` 变量，以匹配您想要可视化的日志文件（例如，`algo_name = 'ppo'`, `env_name = 'env4'`）。
3. **生成图表**:
   ```bash
   python plot.py
   ```
   该脚本会自动读取对应的 `_log.csv` 和 `_rollout.csv` 文件，并生成奖励曲线、航行轨迹等图像，保存在项目根目录下。

## 🤝 贡献

我们欢迎任何形式的贡献！如果您有任何建议或发现了 bug，请随时提出 Issue 或提交 Pull Request。

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。
