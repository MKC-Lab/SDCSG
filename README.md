﻿# 立场驱动的组合属性图提示的可控文本生成 (Stance-Driven Controllable Statement Generation via Compositional Attribute Graph Prompting with LLMs)

提出了一种立场驱动的组合属性图提示的可控文本生成的方法，采用组合属性图来识别与立场属性维度相关的语义结构和关键词，引导大模型可以做更好的生成。本项目实现了多种受控文本生成方法，包括基于语义属性图（SD-CPAG）、词级控制和其他基线方法（FUDGE、PREADD）等，支持多模型、多任务的立场文本生成实验。

## 目录结构
- `main.py`：主程序入口，支持不同模型和任务的批量生成与结果保存。
- `analyst.py`：结果分析与统计脚本。
- `test.py`：实验测试脚本，便于快速评估方法效果。
- `config.py`：全局配置，包括模型路径、任务配置等。
- `convert.py`：数据格式转换工具。
- `method/`：各方法实现
  - `fudge/`：FUDGE 控制方法
  - `preadd/`：PREADD 控制方法
  - `sdcpag/`：语义属性图方法（SD-CPAG 及其变体）
- `train/`：分类器训练相关代码
- `utils/`：工具函数与评测指标
- `requirements.txt`：依赖包列表

## 快速开始

1. 安装依赖
   ```sh
   pip install -r requirements.txt

2. 运行主程序（示例）
    ```python
   python main.py --model_name llama3_8B_Base --task_name 2Positive

3. 方法简介
    SD-CPAG：结合词级和语义级属性图进行立场控制文本生成。
    SemanticOnly：仅用语义级属性图进行生成。
    WordOnly：仅使用词级属性图进行生成。

4. 结果分析
    生成结果保存在 results/ 目录下。


