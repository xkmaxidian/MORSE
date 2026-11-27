# MORSE：两个数据集的完整主流程

这份代码包括 **PDAC A-1** 和 **Breast Cancer Visium Block A Section 1**。
每个数据集均提供计数、坐标、配套 H&E 图像、参数文件、命令行入口和逐步 Notebook。
两个数据集共用完整 MORSE 模型，不包含消融或对比算法。

## 安装

建议使用 Python 3.10，在解压后的 `MORSE` 目录执行：

```bash
python -m venv .venv
# Linux：source .venv/bin/activate
# Windows PowerShell：.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

默认安装 CPU 版本；GPU 环境说明见 `docs/ENVIRONMENT.md`。

## 两种使用方式

**直接运行完整流程：**

```bash
python scripts/run_pdac.py
python scripts/run_visium_breast.py
```

配置中均为项目相对路径，无须填写原服务器路径。默认 5,000 轮、3,000 个基因。
可修改 `configs/pdac.json`、`configs/visium_breast.json`，或在命令中使用
`--device cuda:0`、`--epochs`、`--seed`、`--output-dir`。

**逐步查看每个环节：**

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab
```

打开 `notebooks/01_pdac.ipynb` 或 `notebooks/02_visium_breast.ipynb`。
每一步均有说明，依次展示读取数据、归一化/选基因、提取图像特征、训练、生成新点和保存。
Notebook 与命令行调用相同源码，不是另外写一套隐藏实现。

## 源码怎么读

- `src/morse/pipeline.py`：完整流程及调用顺序。
- `data.py`、`preprocessing.py`：数据对齐、计数归一化和基因选择。
- `image_features.py`：15 × 15 图像块的 RGB 均值和标准差，共六维。
- `model.py`：编码器 E、解码器 D、拓扑网络 f、形变网络 phi。
- `losses.py`、`training.py`：五项损失、KNN 图、训练和推断。
- `locations.py`：新点候选、去重和坐标范围筛选。
- `output.py`、`visualization.py`：保存文件、可选绘图。

更详细的参数、数据来源、方法对应关系和源码整理记录在 `docs/` 中。

## 数据与输出的区别

包内 `data/` 是输入：PDAC 有 428 个已配准实测点，乳腺癌 Visium 有 3,798 个组织内实测点。
输入使用计数矩阵，避免把已经 log 归一化的 h5ad 再次归一化。

代码运行后才会创建 `outputs/`。其中 `observed_expression.h5ad` 保存预处理后的实测值，
`generated_expression.h5ad` 保存所有实测位置与新位置的模型预测。
`is_measured_location` 标记位置类型，表达单位为归一化后的 log1p 值，不是整数计数。
还会保存基因列表、点坐标、训练损失、模型和实际运行配置，方便别人继续使用。
默认不覆盖已有输出目录。

新点经过凸包、最近邻距离和去重筛选；这不是显式组织掩膜，不能保证剔除每个组织空洞。
实际生成倍数记录在运行清单中，可能少于四倍。PDAC 图像是原项目配准后的工作图；
Visium 使用官方 hires 图及对应缩放比例。具体配准规则见 `docs/DATA.md`。

本版本保留原 MORSE 完整分支的网络结构，并按正文公式整理目标函数。
与历史源文件的差异逐项写在 `docs/SOURCE.md`，不能把它说成未改动的历史归档。
本次只整理文件并做静态检查，没有执行训练、推断或 Notebook；包内没有运行结果、权重、
消融实验、对比实验或单项测试。
