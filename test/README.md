# AI 原油价格预测项目 (Citi-Plus)

> [!NOTE]
> 本项目采用混合深度学习架构 (Transformer + BiLSTM + CNN) 结合市场新闻情绪分析。

简单构想：

1. 总体是两个东西：后端+前端，后端是个AI模型，前端是个网页，网页实在不行让AI写一写，我也学一学
2. 模型的输入数据是各种油价的影响因素+前若干天的油价+一个新闻影响因素
3. 训练的时候直接从官方接口里获取各种数据，利用 VIX (恐慌指数) + 价格动向 算出一个当时的新闻对于油价的影响
4. 模型的输出有三个：
   * （1）预测出的下一天的油价（一个区间）
   * （2）预测油价的置信度
   * （3）用SHAP去黑盒化，体现出每个因素对油价影响的占比
5. 推理的时候准备用爬虫去网上爬一下新闻数据，可以只把新闻的标题取下来，用一个LLM比如DeepSeek的API，生成每条有关的新闻对油价的影响，负面冲击还是正面冲击（-1~1），传给模型

---

项目使用深度学习模型 (**Hybrid Transformer-LSTM**: 混合 Transformer 和 LSTM) 来预测 WTI 原油每日收盘价格。项目包含完整的数据加载、特征工程、模型训练、验证和可视化流程。

**当前情况 (2026.01.19)**：
- 修复了因为"未来数据泄露"导致模型看起来很准但在新数据上失效的问题。
- 修复了预测值为 0 的 Bug（因反归一化逻辑错误导致）。
- 模型架构升级为多尺度卷积 + 时间注意力机制，解决了预测滞后问题。
- 新闻爬虫已通过测试 (DuckDuckGo + Investing.com)。

## 目录结构

*   `README_ENV.md`: 配置所需的运行环境的详细教程（使用conda）
*   `config.py`: 项目的全局配置文件，包含所有参数设置。
*   `data_loader.py`: 负责数据的读取、清洗、API下载（如 Yahoo Finance）、技术指标计算等特征工程。
*   `train.py`: 训练脚本，定义了模型训练循环、损失函数 (MSE + 敏感度损失) 和早停机制。
*   `main.py`: 推理与分析脚本，用于加载训练好的模型，在测试集上进行预测，并生成详细的分析图表。
*   `model.py`: 定义神经网络结构 (`OilPriceTransformer`)。
*   `news_agent.py`: **[新增]** 新闻爬虫与情感分析模块 (DuckDuckGo + DeepSeek API)。
*   `utils.py`: 通用工具函数（如设备检测、随机种子设置）。
*   `input_data/`: 存放本地 CSV 数据的文件夹。
*   `models/`: 存放训练好的模型权重 (`.pth`) 和归一化器 (`.pkl`)。
*   `data/`: 存放处理后的缓存数据。

## 使用方法

### 1. 准备数据

您可以将历史数据 CSV 文件放入 `input_data` 文件夹（文件名如 `CL=F.csv`）。如果没有本地文件，程序会自动尝试从 Yahoo Finance 下载。

> [!WARNING]
> 首次运行时，脚本会生成 `data/oil_data_merged.csv` 缓存。如果修改了特征工程逻辑，请务必手动删除该缓存文件以强制重新生成！

### 2. 训练模型

运行以下命令开始训练：
```powershell
python train.py
```
训练过程会自动保存验证集 Loss 最低的模型到 `models/best_oil_price_model.pth`。

### 3.与新闻结合的推理

运行以下命令生成预测图表：
```powershell
python main.py
```
这将生成以下图片：
*   `oil_price_prediction_full.png`: 仅展示**测试集**（未知数据）的预测结果，包含置信区间和置信度评分。
*   `prediction_analysis.png`: 最近200天的详细回测图。
*   `feature_importance.png`: 必须特征的重要性排行。

> [!TIP]
> 如果想要测试实时新闻对预测的影响，请在 `news_agent.py` 中填入 DeepSeek API Key 并取消注释相关代码。

## 参数详解 (config.py)

您可以在 `config.py` 中修改以下重要参数来调整模型行为。

### 数据相关 (`Data Related`)
*   `START_DATE / END_DATE`: 设定训练和回测数据的时间范围。
*   `TICKER_OIL`: 目标预测资产的代码（默认为 `CL=F` 即 WTI 原油）。
*   `REMOVE_EXTREME_OUTLIERS`: **极端值处理开关**。
    *   `True`: 将负油价（如2020年4月的-40美元）视为数据缺失并利用前值填充。**推荐开启**，能让模型学习正常市场规律。
    *   `False`: 保留负油价。

### 特征与模型 (`Feature & Model`)
*   `SEQ_LENGTH (30)`: **序列长度**。模型每次“回头看”过去30天的数据。
*   `HIDDEN_DIM (128)`: **隐藏层维度**。
*   `DROPOUT (0.15)`: **随机失活率**。

> [!CAUTION]
> 随意修改 `SEQ_LENGTH` 后必须删除 `data/` 下的缓存文件并重新运行训练，否则会报错。

### 训练参数 (`Training`)
*   `BATCH_SIZE (64)`: **批次大小**。
*   `EPOCHS (200)`: 最大训练轮数。
*   `LEARNING_RATE (0.0005)`: **学习率**。

## 常见问题处理

1.  **出现 `Dimension mismatch` 错误**
    *   通常是因为更改了特征数量但没有重新训练模型。请先运行 `train.py`。
    
2.  **预测结果为一条直线**
    *   可能是因为 `REMOVE_EXTREME_OUTLIERS` 未开启导致数据归一化异常，或学习率过大导致梯度爆炸。

3.  **爬虫抓不到数据**
    *   爬虫依赖 DuckDuckGo 搜索 `site:investing.com`。如果网络不通畅，请检查 VPN 设置。
