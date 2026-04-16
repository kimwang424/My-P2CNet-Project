🌊 P2CNet 水下图像增强：复现与改进记录

### 📖 项目简介
本项目基于 P2CNet 算法，旨在对水下模糊、偏色的图像进行色彩补偿与清晰度修复。本项目记录了一名“AI 学习者”从零开始在本地环境复现该论文成果的全过程。

### 🛠️ 环境配置 (Environment)
- **操作系统**: Windows 11
- **开发工具**: PyCharm 2024
- **Python 版本**: 3.11 (虚拟环境 .venv)
- **核心依赖**:
  - `torch >= 2.0.0` (已适配 CPU 运行)
  - `numpy == 1.26.4`
  - `kornia`, `timm` (补充安装了原作者缺失的依赖包)

### 🚀 核心改进 (Growth & Fixes)
在复现过程中，我对原代码进行了以下关键改进，使其更加稳定和易用：

1. **CPU 自动兼容补丁**
   - 解决了 `AssertionError: Torch not compiled with CUDA enabled` 报错。
   - 引入了 `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 逻辑。
   - 将所有 `.cuda()` 替换为 `.to(device)`，确保在没有显卡的电脑上也能顺畅运行。

2. **模型加载格式转换**
   - 在加载 `.pth` 权重文件时增加了 `map_location=torch.device('cpu')`，成功解决了跨设备序列化加载的 `RuntimeError`。

3. **高清原尺寸输出优化**
   - 修复了模型输出图片长宽（H, W）缩放的问题。
   - 引入了 `Image.Resampling.LANCZOS` 重采样算法，确保处理后的水下图像能够自动恢复至原始分辨率，方便进行 1:1 的视觉对比。

### 📂 运行说明
1. 将待处理照片放入 `./demo` 文件夹。
2. 运行 `test.py`。
3. 处理后的高清图像将自动保存至 `./results` 文件夹。

### 🌟 感悟
科研之路不分早晚，从第一个 `ModuleNotFoundError` 到最终看到清澈的海底图像，每一步报错都是成长的基石。

### 🚀 级联优化效果对比 (Optimization Matrix)

| 处理阶段 | 技术手段 | 视觉表现 |
| :--- | :--- | :--- |
| **原始输入 (Raw)** | 无 | 严重偏色、水下散射、细节丢失。 |
| **P2CNet 增强** | 物理先验色彩补偿 | **颜色精准**，但因 256px 限制导致画面模糊。 |
| **Real-ESRGAN** | 对抗生成网络 (GAN) | **极致清晰**，但存在塑料感伪影及比例失调。 |
| **最终融合版** | **比例纠正 + 纹理回春** | **最优平衡**：比例回归自然，保留了真实生物纹理。 |

> **⚠️ 硬核修复记录 (Hacker Fix)**
> 由于 PyTorch 环境过新，本项目手动修改了虚拟环境下的 `basicsr/data/degradations.py` 第 8 行，将 `functional_tensor` 修正为 `functional`，成功解决了 RTX 50 系列显卡的兼容性问题。

References：
P2CNet: A P2CNet algorithm for underwater image enhancement.
Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.




