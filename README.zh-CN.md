# API-TAIL: Detecting Model Substitution in LLM APIs through Temperature-Aligned Likelihood Tails

[![arXiv](https://img.shields.io/badge/arXiv-2610.36801-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.36801)

[English](README.md) · [论文](https://arxiv.org/abs/2610.36801) · [实际审计指南](docs/QUICKSTART.md) · [数据格式](docs/SCHEMAS.md) · [验证结果](docs/VALIDATION_REPORT.md)

**当前版本：0.1.1。** API-TAIL 用于检测 LLM API 可能存在的模型替换：使用声明的参考模型为回复评分，并在允许的解码配置范围内校准检测阈值。

## 先选择你要做什么

| 目标 | 需要准备 | 从哪里开始 |
|---|---|---|
| 先跑通软件 | Python 3.10+，包含 pip 和 venv | 下面的 CPU 快速开始 |
| 重新计算论文数值结果 | 同上；仓库已包含所需数值资产 | 下面的 CPU 快速开始 |
| 检测自己的 API | 参考模型、校准/测试提示、合适的硬件和 API 账户 | [实际审计指南](docs/QUICKSTART.md) |

CPU 快速开始在安装完成后可离线运行，无需 GPU、模型权重或 API 密钥。论文回放使用仓库中保存的统计量。

## 1. 下载代码并进入目录

使用 Git 克隆：

```bash
git clone https://github.com/TokenRhythm/API-TAIL.git
cd API-TAIL
```

也可以下载仓库 ZIP 并解压，然后在含有 `pyproject.toml` 的文件夹中打开终端。以下命令都从这个目录执行。

## 2. 安装并跑通 CPU 示例

根据操作系统选择下面的一组命令。

**Linux / macOS / WSL — Bash**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m apitail --version
python -m apitail doctor
python -m apitail demo --output work/demo1
python scripts/reproduce_paper.py --output work/paper1
```

**Windows — PowerShell**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m apitail --version
.\.venv\Scripts\python.exe -m apitail doctor
.\.venv\Scripts\python.exe -m apitail demo --output work/demo1
.\.venv\Scripts\python.exe scripts/reproduce_paper.py --output work/paper1
```

如果找不到 `py`，将 Windows 第一条命令换为 `python -m venv .venv`。后续命令直接调用虚拟环境中的 Python，无需运行激活脚本。基础安装只安装 NumPy，因此 `doctor` 显示 `torch: null`、`transformers: null` 属于正常情况。

## 3. 确认运行成功

- 版本命令输出 `apitail 0.1.1`。
- 示例输出 `"synthetic": true`，并保存校准、审计和评估结果。
- 论文回放输出 `"total_checks": 2020`、`"failures": 0`。预期状态为 `passed_with_float32_differences`；这些已记录的精度差异见[复现说明](docs/REPRODUCTION_STATUS.md)。

| 输出文件 | 内容 |
|---|---|
| `work/demo1/summary.json` | 合成示例汇总 |
| `work/demo1/audit.json` | 示例批次的审计结论 |
| `work/paper1/replay.json` | 论文结果核对记录 |
| `work/paper1/main_results.csv` | 重算的方法比较结果 |
| `work/paper1/power_grid.csv` | 重算的功效网格 |
| `work/paper1/pool_ablation.csv` | 校准池消融结果 |

再次验证时，只需重跑示例和回放两条命令，并换用尚不存在的输出目录，例如 `work/demo2` 和 `work/paper2`。示例不会覆盖已有目录；其中的合成数据结果只用于验证程序流程。

## 4. 检测自己的 API

接下来阅读[实际审计指南](docs/QUICKSTART.md)（英文）。指南从安装完成后开始，不会要求重新运行上面的示例。命令使用 Bash；Windows 用户可在 WSL 中另建 Linux 环境运行。

需要准备：

- **声明参考模型**的本地权重，以及兼容的 tokenizer 和聊天模板。
- 彼此独立的校准提示集和测试提示集。指南每组至少使用 400 个不同提示，审计批次预算为 200。
- 能加载参考模型的硬件。已测试的 27B 模型约需 50 GiB 显存，具体要求随模型而变。
- 真实 API 的端点地址、模型标识和密钥。

执行顺序：

```text
生成合法配置下的校准回复 → 参考模型评分 → 校准阈值
采集 API 回复 → 使用同一参考模型评分 → 审计
```

指南给出创建配置、设置密钥、预览请求、执行采集和查看结论的完整步骤。采集默认只显示请求计划，添加 `--execute` 才会发送请求。替代模型只用于最后可选的功效实验。

## 进一步使用

- [数据格式与数值约定](docs/SCHEMAS.md)
- [方法、变体和基线接口](docs/DESIGN_MAP.md)
- [资产与复现范围](docs/ASSETS.md)
- [安装环境](docs/ENVIRONMENT.md)
- [测试及扩展方式](CONTRIBUTING.md)

[验证报告](docs/VALIDATION_REPORT.md)记录了：安装模型后端后 78 项测试通过；轻量 CPU 环境中 64 项通过、14 项可选测试跳过；2,020 项论文结果核对、1,280 次原函数对照，以及本地 GPU 冒烟测试。

## 许可证与引用

代码及随包提供的数值统计量采用 [Apache-2.0](LICENSE)，版权人为 TokenRhythm Technologies。模型权重和原始提示/回复文本是独立资产，其获取与许可范围见[资产说明](docs/ASSETS.md)。

论文已发布在 arXiv：[arXiv:2610.36801](https://arxiv.org/abs/2610.36801)。使用 API-TAIL 时请引用该论文；[CITATION.cff](CITATION.cff) 中有相同的引用元数据。

```bibtex
@misc{tokenrhythm2026apitail,
  author = {{TokenRhythm Technologies}},
  title = {{API-TAIL}: Detecting Model Substitution in {LLM APIs} through Temperature-Aligned Likelihood Tails},
  year = {2026},
  eprint = {2610.36801},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2610.36801}
}
```
