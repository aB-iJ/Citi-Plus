# Citi-Plus

## 2026.01.19 02:35 , 李桂聿 - 添加了test文件夹 和 .gitignore文件
- Citi-Plus/test/README.md文件描述了对于整个项目的想法、当前已完成的部分、遇到的问题 和 当前模型的部署方法
- Citi-Plus/test/README_ENV.md文件详细给出了配置环境所需要的所有步骤（前提是你安装了conda）
- Citi-Plus/.gitignore文件规定了不需要git追踪的文件/文件夹，以后对这些文件/文件夹所做的所有更改不会被git跟踪，也不会被push到GitHub里。
> [!CAUTION]
> **请不要随便删除/修改/移动 Citi-Plus/.gitignore ，否则整个虚拟环境和缓存文件夹都将被git跟踪，轻易无法删除！**

## 2026.01.19 03:45 , 李桂聿 - 更改了.gitignore文件使git跟踪 *.csv 数据文件
- `Citi-Plus/.gitignore`文件里删除了 *.csv，使git跟踪 *.csv 文件
- `Citi-Plus/test/input_data`下的五个csv文件被跟踪
- `Citi-Plus/test/data`下的一个csv文件被跟踪

## 2026.01.19 12:19 , 李桂聿 - 优化了.md文件的代码使其正确的在GitHub上渲染
- `Citi-Plus/README.md`
- `Citi-Plus/test/README.md`
- `Citi-Plus/test/README_ENV.md`