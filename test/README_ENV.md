# 环境配置指南 (Environment Setup)

为了运行本项目的油价预测模型，请按照以下步骤配置 Python 环境。

## 1. 创建 Conda 虚拟环境

```powershell
conda create -n citi_oil_ai python=3.10
conda activate citi_oil_ai
```

## 2. 安装依赖包

请根据你的硬件选择合适的 PyTorch 版本。

### 2.1 基础依赖
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn yfinance akshare ta shap tqdm textblob
```

### 2.2 PyTorch (AI 模型框架)

**选项 A: NVIDIA GPU (CUDA)**
如果你的设备有 NVIDIA 显卡，请安装 CUDA 版本 (以 CUDA 11.8 为例):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**选项 B: Intel ARC 显卡 / DirectML 支持**
如果你使用的是 Intel ARC 显卡或其他支持 DirectML 的设备:
```powershell
pip install torch-directml
```
> [!NOTE]
> 注意: 安装 torch-directml 时如果不兼容，可直接安装 CPU 版 torch，代码会自动降级。*

**选项 C: 仅 CPU**
```powershell
pip install torch torchvision torchaudio
```

### 2.3 其他工具 (XGBoost 等)
```powershell
pip install xgboost lightgbm
```

## 3. 运行项目

1. **获取数据与清洗**: 运行 `data_loader.py` (通常会在 `train.py` 中自动调用，也可单独测试)
2. **训练模型**:
   ```powershell
   python train.py
   ```
3. **预测与解析 (SHAP)**:
   ```powershell
   python main.py
   ```

## 4. 已使用的关键包列表

*   `yfinance`: 获取全球 WTI 原油、标普500、美元指数等历史数据。
*   `akshare`: 获取国内或其他特定的宏观经济数据补充。
*   `pandas / numpy`: 数据清洗、处理、归一化 (-1 到 1)。
*   `torch (PyTorch)`: 构建复杂的 LSTM/Transformer 时序预测模型。
*   `shap`: 模型去黑盒化，解释各因子的权重。
*   `ta`: 计算 RSI, MACD 等技术指标作为输入特征。
*   `textblob`: 用于新闻情感分析 (NLP) 初步处理。
