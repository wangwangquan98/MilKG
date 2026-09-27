# MilKG-QA 论文方法复现

本目录实现 `MilKG.pdf` 所述的 **MilKG-QA: Knowledge Graph-Driven Synthetic SFT Dataset Generation for the Military Domain via Rule-Based Traversal** 的数据生成流程。`GraphGen/` 仅供对照；本实现是独立 Python 包，不导入 GraphGen。另提供简洁的 Vue 本地工作台。

## 论文与 GraphGen 的区别

| 环节 | GraphGen 参考项目 | 本复现中的 MilKG-QA |
| --- | --- | --- |
| 本体 | 通用实体关系抽取 | 14 类军事实体、12 类有方向的关系及类型约束 |
| 模型 | 可用单模型贯穿流程 | 低温抽取模型和高温生成模型分别配置，默认温度 0.1/0.7 |
| 子图组织 | ECE 等统计排序及通用分区 | 装备链、对抗链、战役全景、实体比较、2–4 跳推理五种规则 |
| 问答 | GraphGen 的通用生成器 | 单选、多选、CoT 简答、判断、填空；按深度标注难度 |
| 质量控制 | 通用过滤器 | 本体/证据校验、选项检查、KG 来源边检查、中文字符级 ROUGE-L 去重、长度限制 |

论文把 GraphGen 风格基线描述为随机边选择。这里没有把 GraphGen 的 ECE 分区器复制到 MilKG 主流程中。

## 安装

Python 3.10+：

```powershell
python -m pip install -e ".[documents,test]"
python -m pytest -q
```

### 启动本地 WebUI

需安装 Node.js 22.18+ 与 npm。首次启动：

```powershell
python -m pip install -e ".[documents,web]"
cd frontend
npm ci
npm run build
cd ..
$env:ALIYUN_API_KEY = "你的 API Key" # 已设置环境变量时可跳过
python -m milkg.web
```

打开 `http://127.0.0.1:8000`。开发页面时可在另一终端执行 `cd frontend; npm run dev`，访问 Vite 显示的地址；`/api` 会代理到本地后端。WebUI 支持上传 TXT、MD、PDF、DOCX（上限 20 MB）、分别选择抽取与生成模型及温度、查看分块/抽取/建图/生成阶段进度与错误、实时预览校验通过的问答，并下载 Alpaca、ShareGPT 或 ChatML JSON。API Key 可在页面临时填写，也可由后端读取 `ALIYUN_API_KEY`；不会保存到任务配置或数据集。网页仅在本机 `127.0.0.1` 监听。任务中间结果位于 `output/webui/<任务 ID>/`，该目录不纳入 Git。

默认模型分别为 `qwen3.5-flash` 和 `qwen3.5-plus`，默认端点为阿里云百炼北京兼容接口。模型可在页面改为当前账号有权调用的其他 Qwen 模型；端点应与 API Key 所在地域匹配。默认温度分别为 0.1 和 0.7。界面的预览会显示来源边 ID 便于审查；生成的自由文本仍需人工核验。

核心只依赖 NetworkX；PDF 与 DOCX 读取分别使用可选的 `pypdf`、`python-docx`。测试范围限定为本项目的 `tests/`，不会收集 `GraphGen/` 测试。

## 离线检查五种遍历

随附的 `examples/fictional_extractions.json` 使用虚构实体，不代表真实军事事实。

```powershell
python -m milkg.cli build --extractions examples/fictional_extractions.json --output output/demo
python -m milkg.cli traverse --output output/demo
python -m milkg.cli evaluate-traversal --output output/demo
```

检查 `output/demo/graph.json` 和 `output/demo/subgraphs.json`。`--extractions` 适用于**已审核、可信的结构化输入**，跳过原文证据核验；从原始文档运行时则逐块核验抽取的实体名称与关系证据是否出现在原文中。
`traversal_metrics.json` 汇总子图数量、结构类型、实体类型覆盖和深度，并提供同起点同深度的随机选边基线。它对应论文表 III 对基线的文字描述，不是 GraphGen 原始 ECE 计算器。

## 从文档生成 SFT 数据

抽取与生成端点均使用 OpenAI 兼容的 `/chat/completions` 协议，可以分别指向本地服务和远程服务。例子中的端点和模型名需要替换成实际可用值：

```powershell
$env:MILKG_EXTRACT_URL = "http://localhost:8000/v1"
$env:MILKG_EXTRACT_MODEL = "YOUR_FINE_TUNED_EXTRACTION_MODEL"
$env:MILKG_GENERATE_URL = "https://YOUR_ENDPOINT/v1"
$env:MILKG_GENERATE_MODEL = "YOUR_GENERATION_MODEL"
$env:MILKG_GENERATE_KEY = "YOUR_API_KEY"
python -m milkg.cli run .\documents --output output/run --format alpaca
```

也可以分阶段运行：

```powershell
python -m milkg.cli build .\documents --output output/run
python -m milkg.cli traverse --output output/run
python -m milkg.cli generate --output output/run --format sharegpt
```

支持 TXT、Markdown、PDF、DOCX。输出包括 `chunks.json`、`extractions.json`、`graph.json`、`subgraphs.json`、`qa.json`、`stats.json` 和 `sft_alpaca.json` / `sft_sharegpt.json` / `sft_chatml.json`。`qa.json` 保留题型、来源边 ID、实体 ID、难度和证据元数据；SFT 文件可用于 LLaMA-Factory。部分题目被过滤的原因计入 `stats.json`。

常用参数：`--max-chars 1800 --overlap 180` 控制文档分块；`--min-confidence` 控制抽取门槛；`--max-subgraphs` 和 `--max-overlap` 控制子图数量与节点重叠；`--per-subgraph` 控制每个子图尝试生成几题；`--question-types` 可选 `single_choice,multiple_choice,cot,true_false,fill_blank`。默认值是可配置的工程设置，论文没有给出这些参数的完整数值。

同义词词典是 JSON 对象，键为别名、值为规范名，例如 `{"甲型平台":"甲平台"}`，通过 `--synonyms` 传入。建图同时使用同类型编辑距离匹配，并以冲突属性阻止可疑合并。

### `test_input.txt` 的实际 API 运行

本次使用环境变量 `ALIYUN_API_KEY`，通过阿里云百炼北京兼容端点调用 `qwen3.5-flash` 抽取、`qwen3.5-plus` 合成；未把密钥写入文件。命令如下：

```powershell
$env:MILKG_EXTRACT_KEY = $env:ALIYUN_API_KEY
python -m milkg.cli build test_input.txt --output output/test_input_qwen --extract-url https://dashscope.aliyuncs.com/compatible-mode/v1 --extract-model qwen3.5-flash --disable-thinking
python -m milkg.cli augment-specs --output output/test_input_qwen
$env:MILKG_GENERATE_KEY = $env:ALIYUN_API_KEY
python -m milkg.cli generate --graph output/test_input_qwen/graph_augmented.json --output output/test_input_qwen --generate-url https://dashscope.aliyuncs.com/compatible-mode/v1 --generate-model qwen3.5-plus --disable-thinking --include-atomic --per-subgraph 1 --format alpaca
```

抽取后仅有 6 条关系通过严格证据与本体校验。`augment-specs` 从相邻原文中的装备名称和属性值补出 13 条有原文证据的 `Weapon-Spec` 边；`--include-atomic` 从关系边切出单跳“简单”题。二者是本实现为知识图谱较稀疏的文献所做的明确补充，不改变五种主要遍历规则。端点若因地域或工作空间不同而不能使用，需要换成与 API Key 同地域的地址；阿里云推荐工作空间专属域名。

这次运行共尝试 32 题，自动过滤 18 题，最终得到 **14 条**经过来源检查和人工修订的 Alpaca 记录，覆盖五种题型。最终文件为 `output/test_input_qwen/sft_alpaca.json`；`qa.json` 保留图谱溯源字段；`quality_report.json` 记录输入 SHA-256、模型、图谱规模和题型统计。`qa_uncurated.json` 与 `curation_report.json` 保存修订前内容和每项改动。`examples/curate_test_input.py` 仅针对这次已保存的模型输出，不适用于任意新生成文件。

## 抽取模型训练与评估接口

论文使用人工标注的军事 NER/RE 数据微调轻量抽取模型，但没有公开所述 20,000+ 实体提及、12,000 关系实例的标注文件。本项目提供转换和精确匹配 F1 计算。标注数组中的每项需有 `text`、`entities`、`relations`；关系需有原文中的连续 `evidence`。结构可参考 `tests/test_milkg.py` 的 `test_extractor_training_and_metrics`。

```powershell
python -m milkg.cli prepare-extractor --annotations data/train_annotations.json --output output/train
python -m milkg.cli evaluate-extractor --annotations data/test_annotations.json --predictions data/test_predictions.json --output output/eval
python -m milkg.cli evaluate-qa --annotations data/benchmark_answers.json --predictions data/model_answers.json --output output/eval
```

前者输出 `extractor_sft_alpaca.json`，可用 `examples/llamafactory_extractor.yaml` 在 LLaMA-Factory 中微调。`examples/llamafactory_downstream.yaml` 展示论文的 3 epoch LoRA 下游训练对接。模型路径、数据路径、批量和学习率由实际硬件与数据决定；论文未完整报告这些超参数，不应把示例配置误当成作者原始配置。
`evaluate-qa` 需要带相同 `id` 的标准答案和模型预测数组，报告总体及各题型的精确准确率和字符级 ROUGE-L；论文的 BERTScore 与人工评估需要另行准备模型和评审数据。

## 实现细节与边界

- **本体约束：**论文只明确举出 `Equip-Carry` 的来源/目标类型，其余类型约束在 `milkg/ontology.py` 依据表 I 语义显式补全；例如 `Equip-Develop` 的研发机构在 14 类实体中没有专门类型，当前映射到组织或设施类。这是实现解释，不是论文给出的完整原始 schema。
- **遍历：**路径评分为类型多样性 0.4、2–3 跳偏好 0.25、平均关系置信度 0.35。权重是可检查的工程选择，论文只说明三项加权而未给权重。默认过滤节点重叠达 50% 的候选，并优先保留每种有候选的语义规则至少一条路径，避免通用 BFS 抹去专用规则。
- **真假题：**错误陈述以同类型目标实体替换一条原事实，并确认替换后的关系不存在于当前 KG。这是**封闭世界近似**，图中不存在并不严格证明现实中为假，需要人工审查。
- **自动核验：**代码验证结构、来源边、声明的实体、正确选项与 KG 实体名、同类型干扰项、错误题替换、长度和 ROUGE-L 重复。自由文本的隐含事实或未声明实体无法仅靠字符串规则完全识别，正式发布前应人工抽检或增加独立事实核验模型。
- **调用可靠性：**在线生成逐题保存检查点；网络超时和服务端临时错误会重试，`--resume` 可从同一配置下的检查点继续。对千问 JSON 输出可使用 `--disable-thinking`。
- **论文内部不一致：**第 III-C 节将生成模型举例为 Qwen2.5-72B，第 IV-A 节实际写 DeepSeek-V4-Pro；第 IV-C 节写 Qwen3-8B，而表 V 标为 Qwen2.5-7B-Instruct。本代码不硬编码这些有冲突的模型名称。
- **实验数值：**仓库没有论文的 40 篇文档、标注集、微调模型和 200 题基准，因此不能重现论文表 II–V 的 F1、主观评分及下游准确率。当前测试验证方法流程与数据格式，不声称复算论文统计结果。
