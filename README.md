# Citi-Plus

## 2026.01.19 02:35 , 李桂聿 - 添加了`Citi_Plus/test`文件夹 和 `.gitignore`文件
- `Citi-Plus/test/README.md`文件描述了对于整个项目的想法、当前已完成的部分、遇到的问题 和 当前模型的部署方法
- `Citi-Plus/test/README_ENV.md`文件详细给出了配置环境所需要的所有步骤（前提是你安装了conda）
- `Citi-Plus/.gitignore`文件规定了不需要git追踪的文件/文件夹，以后对这些文件/文件夹所做的所有更改不会被Git跟踪，也不会被push到GitHub里。
> [!CAUTION]
> **请不要随便删除/修改/移动 `Citi-Plus/.gitignore` ，否则整个虚拟环境和缓存文件夹都将被git跟踪，轻易无法删除！**

## 2026.01.19 03:45 , 李桂聿 - 更改了`.gitignore`文件使git跟踪`*.csv`数据文件
- `Citi-Plus/.gitignore`文件里删除了`*.csv`，使git跟踪 `*.csv` 文件
- `Citi-Plus/test/input_data`下的五个csv文件被跟踪
- `Citi-Plus/test/data`下的一个csv文件被跟踪

## 2026.01.19 12:19 , 李桂聿 - 优化了.md文件的代码使其正确的在GitHub上渲染
- `Citi-Plus/README.md`
- `Citi-Plus/test/README.md`
- `Citi-Plus/test/README_ENV.md`

## 2026.01.19 15:30 , 李桂聿 - 修复数据泄露与模型升级
- **修复**: 解决了 `data_loader.py` 中的未来数据泄露问题（移除了 Oracle 信号）。
- **升级**: 引入了混合模型架构 (Transformer + LSTM + CNN)，增加了时间注意力机制。
- **功能**: 更新了 `news_agent.py`，支持使用 DuckDuckGo 搜索 `investing.com` 的历史新闻。
- **文档**: 更新了 `test/README.md` 并汉化了部分核心代码注释。
- **git配置** :删除了 `*.pkl`,`Citi-Plus/test/model`下的三个`.pkl`文件被跟踪
> [!WARNING]
> 由于特征工程逻辑变更，请务必删除 `test/data` 下的缓存文件并重新训练模型！

> [!IMPORTANT]
> 模型依然存在较为严重的滞后性，待改进

> [!IMPORTANT]
> DeepSeek的API密钥并未公开，需要在虚拟环境里设置环境变量，方法：
> ```powershell
> $env:DeepSeek_API = "sk-xxxxxxxx"
> ```

## 2026.01.19 21:02 , 李桂聿 - 将`Citi_Plus/README.md`重命名为`History.md`，优化了`.md`文件的代码
- `Citi-Plus/History.md`
- `Citi-Plus/test/README.md`

## 2026.01.19 23:14 , 李桂聿 - 将`Citi_Plus/History.md`重命名为`README.md`，优化了缓存逻辑，极少的使用缓存
- 出于对GitHub首页展示信息的规则的考虑，将`Citi-Plus/History.md`重命名为README.md
- `Citi-Plus\test\data\oil_data_merged.csv`将在每次运行`Citi_Team\Citi-Plus\test\main.py`时重新生成
- 取消了图片`Citi_Team\Citi-Plus\test\oil_price_prediction_full.png`生成所需要数据的缓存机制，防止图片不更新
- 尽力尝试解决了模型滞后性的问题但仍然有严重的滞后性