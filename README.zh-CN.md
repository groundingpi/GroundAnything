# GroudingAnything

[English](README.md) | 简体中文

## 快速开始

在仓库根目录执行命令。`requirements.txt` 安装客户端和安装工具所需的依赖，训练、服务与评测的依赖分别放在 `requirements/` 下。如需通过 `environment.yml` 创建 Conda 环境，参见[环境说明](environments/README.md)。

客户端连接已经启动的模型服务。如果需要自行部署模型，请先完成下方的[推理](#推理)步骤。客户端本身不加载权重，也不依赖 Torch。

```bash
python -m pip install -r requirements.txt
```

```python
from grounding_anything import GroundingAnything, visualize

client = GroundingAnything()
result = client.predict("your_image.jpg", "the red car", task="bbox")
print(result.to_dict())
if result.valid:
    visualize("your_image.jpg", result).save("prediction.png")
```

点定位使用 `task="point"`。返回结果包含原始输出、停止原因和词元用量；格式错误或被截断的响应会返回 `valid=False`。坐标采用 0–999 网格，可视化时映射到图片像素。

图片准备、JSON 输出和可视化命令见[示例说明](examples/README.md)。

## 训练

以下命令均在项目根目录执行。`run.py` 自动选择环境和默认配置；每个环境的安装命令只需执行一次。依赖与安装要求见[环境说明](environments/README.md)。

将基座模型、模型代码和处理器放入 `weights/base_model/`，将完整 DLM 封装模型权重和分词器放入 `weights/dlm/`。按照[数据准备说明](docs/DATA_PREPARATION.md)，在 `data/` 下准备训练样本和数据检查报告。权重和数据需单独准备；项目不提供通用的 JSONL 到训练缓存转换器。

先在通用训练阶段（`general`）的 YAML 中配置模型、数据和分布式参数，然后启动训练：

```bash
python3 run.py setup train
python3 run.py train --config configs/release/general_train.yaml
```

通用训练阶段模板使用 8 节点、每节点 8 个设备，检查点默认保存到 `outputs/general/`。运行前设置集群的协调节点地址和各节点编号。数据要求、四阶段训练及可选 RLV2 后训练见[训练说明](docs/TRAINING.md)。

部署新的检查点时，需要在 `weights/dlm/` 中准备完整 DLM 封装模型权重及匹配的分词器，然后生成新的模型包。不要复用由旧检查点构建的模型包。

## 推理

服务使用随项目提供的定制 SGLang 引擎，默认采用 **DecodeV4 直接分块去噪**，也支持因果自回归和贪心自推测解码。安装环境，并在新的输出目录中准备模型包：

```bash
python3 run.py setup serve
python3 run.py prepare-model
```

启动默认的分块去噪服务：

```bash
python3 run.py serve
```

如需切换解码方式，先停止已有服务，再选择以下其中一种：

```bash
python3 run.py serve --decoder causal
python3 run.py serve --decoder speculative
```

三种解码方式均使用 `weights/dlm_bundle/`。默认地址为 `http://127.0.0.1:8101/v1`，模型 ID 为 `groundinganything`。服务就绪后，在另一个终端运行客户端示例。模型包、自定义配置及原生参考服务见[推理说明](docs/INFERENCE.md)。

## 评测

评测数据路径统一在 [`configs/datasets.yaml`](configs/datasets.yaml) 配置，支持相对 `data_root` 的路径及绝对路径。详见[评测数据说明](docs/EVALUATION.md#data-and-tasks)。

按照[评测说明](docs/EVALUATION.md)，在 `data/eval/` 下准备图片和标注。保持模型服务运行，在另一个终端执行：

```bash
python3 run.py setup eval
python3 run.py eval
```

默认配置为 `configs/eval/dlm.yaml`，按所选任务应用 DecodeV4 五档策略。发送请求前，评测程序会检查 SGLang 服务实际使用的解码方式。如果启动的是其他解码服务，请选择匹配的评测策略：

```bash
python3 run.py eval --decoder causal
python3 run.py eval --decoder speculative
```

这些命令只连接已有服务，不会启动服务或切换其解码方式。因果自回归与自推测评测采用贪心采样，各自使用独立的默认运行 ID。运行前检查模型包、服务地址、模型 ID 和数据路径。推荐的 SGLang 服务使用 `service_contract: openai`。

默认 `limit: 8` 只运行所选任务的前 8 个样本。全量评测需设置 `limit: null`，并为每次运行选择新的 `run_id`。指标保存到 `outputs/eval/<run_id>/summary.json`，原始响应保存到同目录的 `responses.jsonl`。

## 配置与文档

使用 `python3 run.py --help` 查看命令。`--config` 指定自定义启动或评测 YAML，`--venv` 指定其他环境；添加 `--dry-run` 可预览实际执行命令。实际执行时，底层脚本会检查完整配置和所需资源。

| 路径 | 用途 | 说明 |
|---|---|---|
| `run.py` | 安装、训练、推理服务与评测入口 | `python3 run.py --help` |
| `models/` | 训练与推理共用的模型定义 | [推理说明](docs/INFERENCE.md) |
| `train/` | 数据检查、分词器工具、训练与检查点处理 | [训练说明](docs/TRAINING.md) |
| `infer/` | 模型加载、解码与 HTTP 服务 | [推理说明](docs/INFERENCE.md) |
| `eval/` | 任务、提示词、请求与指标计算 | [评测说明](docs/EVALUATION.md) |
| `configs/` | 启动、训练、评测和数据配置 | [数据准备](docs/DATA_PREPARATION.md) |

完整流程通过源码目录中的 `run.py` 执行；也可直接使用底层 `scripts/` 和 YAML 入口。wheel 安装包只包含独立的 `grounding_anything` 客户端。本项目可独立使用，训练和推理服务使用各自的独立环境。

## 当前状态与许可证

支持的使用流程见上文。模型权重和数据集需单独获取。RLV2 是唯一启用的强化学习路线，RLV1/RLV3 支路保持禁用。

项目原创贡献采用 [Apache License 2.0](LICENSE)，项目方不额外附加限制；第三方代码、衍生模型代码及权重仍遵守各自适用的上游许可证。第三方声明、原始许可证及源码来源保留在 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)、`licenses/` 和 `third_party/` 中。
