# MilKG

将 TXT、Markdown、PDF 或 DOCX 文档转换为可预览、可下载的 SFT 问答数据集。项目提供 Vue 本地界面和命令行两种使用方式。

## 环境与依赖

| 用途 | 要求 |
| --- | --- |
| Python 后端与命令行 | Python 3.10 或更高版本 |
| WebUI 构建 | Node.js 22.18 或更高版本、npm |
| 在线生成 | 可调用的 OpenAI 兼容聊天模型 API 与 API Key；默认配置使用阿里云百炼 Qwen |
| 跨文档图谱 | Neo4j 服务及其 Bolt 地址、用户名和密码；本地 JSON 模式无需 Neo4j |

Python 依赖由 `pyproject.toml` 管理：

- 核心：`networkx`。
- 读取 PDF / DOCX：`pypdf`、`python-docx`，对应 `documents` 可选依赖。
- Web 服务：`fastapi`、`uvicorn`、`python-multipart`，对应 `web` 可选依赖。
- Neo4j 存储：官方 `neo4j` Python 驱动，对应 `neo4j` 可选依赖。
- 测试：`pytest`，对应 `test` 可选依赖。

前端依赖由 `frontend/package-lock.json` 锁定，主要是 Vue 3、Vite 和 Vue 的 Vite 插件。只使用命令行时无需安装 Node.js。

## 安装

以下命令适用于 Windows PowerShell：

```powershell
git clone https://github.com/wangwangquan98/MilKG.git
cd MilKG
python -m venv .venv
./.venv/Scripts/python -m pip install --upgrade pip
./.venv/Scripts/python -m pip install -e ".[documents,web,neo4j]"
cd frontend
npm ci
npm run build
cd ..
```

如果只需要命令行，可跳过 `frontend` 的安装与构建，并将 Python 安装命令改为：

```powershell
./.venv/Scripts/python -m pip install -e ".[documents]"
```

## 使用 WebUI

启动本地服务：

```powershell
./.venv/Scripts/python -m milkg.web
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。服务默认只监听本机 `127.0.0.1`。

1. 选择“构图并生成”“只构建图谱”或“只生成 SFT”。构图任务上传 TXT、MD、PDF 或 DOCX 文件，单文件上限为 20 MB；独立生成任务无需上传文件。
2. 填写 API Key，或在启动服务前设置环境变量 `ALIYUN_API_KEY`。页面填写的 Key 只用于当前任务，不写入任务配置或数据集。
3. 选择本地 JSON 或 Neo4j 存储。使用 Neo4j 时填写 Bolt 地址、数据库、用户名和密码；构图选择“新建逻辑图谱”或“扩展已有图谱”。扩展或独立生成时点击“读取图谱列表”并选择目标图谱。
4. 设置当前阶段所需的模型、温度和高级参数，启动任务并查看日志。
5. 生成任务完成后下载 Alpaca、ShareGPT 或 ChatML JSON。只构图任务不生成 SFT，页面会显示图谱 ID，可供下次扩展或独立生成。

Neo4j 密码可留空读取后端环境变量 `MILKG_NEO4J_PASSWORD`；密码不会写入任务配置。新建逻辑图谱会分配新的图谱 ID，不会清空数据库。扩展图谱会将新文档实体与该图谱中已有实体对齐，并累计节点、关系及文档来源。每次任务仍保留一份 `graph.json` 快照供检查，Neo4j 是跨任务共享图谱的来源。

下载的 SFT 记录只含训练所需的题目和自然语言答案，不包含图谱节点/边 ID 或内部溯源元数据。`qa.json` 保留事实 ID 供本地核验；模型解释若出现图谱引用会先尝试改写，仍不合格则不会进入数据集。

“最多语义子图”只是多关系出题素材的数量上限，实际数量取决于图谱中能连成子图的关系。“补充技术规格关系”和“加入单跳事实”是独立开关；“每图生成次数”是模型尝试次数，未通过校验的问答不会进入数据集。

抽取前会清理常见 OCR 空格并保留段落边界，长段落优先按句末分块；每个分块先提取实体，再调用同一提取模型复查 12 类关系。运行日志显示关系候选与保留数量，`extractions.json` 中的 `diagnostics` 记录过滤原因。关系复查会增加每个分块的 API 调用次数。

默认端点为阿里云百炼北京兼容接口，默认提取模型为 `qwen3.5-flash`，默认生成模型为 `qwen3.5-plus`。如果 API Key 属于其他地域或工作空间，请在界面中填写对应端点。任务文件与中间结果保存在 `output/webui/<任务 ID>/`。

修改前端时，可在后端运行的同时另开终端执行：

```powershell
cd frontend
npm run dev
```

开发服务器显示的地址会自动将 `/api` 请求代理到 `http://127.0.0.1:8000`。

## 使用命令行

命令行的抽取模型与生成模型需要分别配置。以下示例使用 PowerShell 环境变量；也可以使用同名命令行参数 `--extract-url`、`--extract-model`、`--extract-key`、`--generate-url`、`--generate-model` 和 `--generate-key`。

```powershell
$env:ALIYUN_API_KEY = "你的 API Key"
$env:MILKG_EXTRACT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:MILKG_EXTRACT_MODEL = "qwen3.5-flash"
$env:MILKG_EXTRACT_KEY = $env:ALIYUN_API_KEY
$env:MILKG_GENERATE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:MILKG_GENERATE_MODEL = "qwen3.5-plus"
$env:MILKG_GENERATE_KEY = $env:ALIYUN_API_KEY
./.venv/Scripts/python -m milkg.cli run ./your_document.txt --output output/run --disable-thinking --max-subgraphs 24 --include-atomic --max-atomic 24 --format alpaca
```

`run` 接受一个或多个文档或目录。上例生成 `output/run/sft_alpaca.json`。如需其他格式，使用 `--format sharegpt` 或 `--format chatml`。命令行不会自动读取 `ALIYUN_API_KEY`；示例中明确将它赋给抽取和生成所需的两个变量。

### Neo4j 跨文档图谱

先准备 Neo4j 服务。已有服务时直接使用其 Bolt 地址；若使用 Docker，可参照 [Neo4j 官方 Docker 指南](https://neo4j.com/docs/operations-manual/current/docker/introduction/)创建持久化卷并启动容器：

```powershell
docker volume create milkg-neo4j-data
docker run -d --name milkg-neo4j -p 7474:7474 -p 7687:7687 -e "NEO4J_AUTH=neo4j/YourStrongPassword" -v milkg-neo4j-data:/data neo4j:2026.09.0
$env:MILKG_NEO4J_URI = "bolt://127.0.0.1:7687"
$env:MILKG_NEO4J_USER = "neo4j"
$env:MILKG_NEO4J_PASSWORD = "YourStrongPassword"
```

下面把两批文档写入同一个逻辑图谱，再单独从完整图谱生成数据集；抽取与生成模型的环境变量沿用上面的示例：

```powershell
./.venv/Scripts/python -m milkg.cli build ./docs/batch1 --store neo4j --graph-action new --graph-name "轻武器资料库" --output output/build1 --disable-thinking
$graphId = (Get-Content output/build1/graph_ref.json -Raw | ConvertFrom-Json).graph_id
./.venv/Scripts/python -m milkg.cli build ./docs/batch2 --store neo4j --graph-action extend --graph-id $graphId --output output/build2 --disable-thinking
./.venv/Scripts/python -m milkg.cli generate --store neo4j --graph-id $graphId --output output/sft --disable-thinking --include-atomic --format alpaca
```

`graph_ref.json` 保存逻辑图谱 ID；`graph.json` 是当次快照。Neo4j 中的 `MilKGEntity` 节点通过 `MILKG_RELATION` 关系相连，关系类别保存在 `relation_type` 属性；`MilKGDocument` 和 `MENTIONED_IN` 保存文档来源。所有对象带 `graph_id`，可按图谱 ID 查询跨文档实体与关系。重复读取相同文档时，文档 ID 由内容哈希确定，来源不会重复累计。

也可运行 `./.venv/Scripts/python -m milkg.cli graphs --store neo4j --output output/graphs` 列出已有图谱。以下 Cypher 可在 Neo4j Browser 中按图谱 ID 查询实体及其文档来源：

```cypher
MATCH (e:MilKGEntity {graph_id: $graph_id})-[:MENTIONED_IN]->(d:MilKGDocument)
RETURN e.name AS entity, e.entity_type AS kind, collect(DISTINCT d.name) AS documents
ORDER BY entity
```

常用参数：

| 参数 | 作用 | 默认值 |
| --- | --- | --- |
| `--max-chars`、`--overlap` | 文档分块长度与重叠字符数 | 1800、180 |
| `--extract-temperature`、`--generate-temperature` | 两阶段的模型温度 | 0.1、0.7 |
| `--min-confidence` | 抽取结果最低置信度 | 0 |
| `--max-subgraphs`、`--max-atomic` | 语义子图与单跳事实数量上限 | 500、30 |
| `--include-atomic` | 额外生成单跳事实题目 | 关闭 |
| `--per-subgraph` | 每个子图尝试生成的题目数 | 1 |
| `--question-types` | 逗号分隔的题型列表 | 全部五种 |
| `--resume` | 从现有 `qa.json`、`stats.json` 检查点继续生成 | 关闭 |

题型值为 `single_choice`、`multiple_choice`、`cot`、`true_false`、`fill_blank`。完整参数可通过 `./.venv/Scripts/python -m milkg.cli -h` 查看。

### 分阶段运行

需要在生成前补充文档中明确出现的技术规格关系时，依次运行：

```powershell
./.venv/Scripts/python -m milkg.cli build ./your_document.txt --output output/run --disable-thinking
./.venv/Scripts/python -m milkg.cli augment-specs --output output/run
./.venv/Scripts/python -m milkg.cli generate --graph output/run/graph_augmented.json --output output/run --disable-thinking --max-subgraphs 24 --include-atomic --max-atomic 24 --format alpaca
```

也可在 `build` 后运行 `traverse`，单独生成 `subgraphs.json`。若已有可信的结构化抽取结果，可用 `build --extractions examples/fictional_extractions.json --output output/demo` 离线建图，无需调用抽取模型。

输出目录中的 `chunks.json` 是文档分块，`extractions.json` 是实体关系抽取结果，`graph.json` 或 `graph_augmented.json` 是知识图谱，`subgraphs.json` 是用于出题的子图，`qa.json` 保留问答及来源事实 ID，`stats.json` 记录通过和过滤数量，`sft_<格式>.json` 是最终数据集。生成的自由文本仍需人工核验。

## 运行测试

```powershell
./.venv/Scripts/python -m pip install -e ".[documents,web,test]"
./.venv/Scripts/python -m pytest -q
```
