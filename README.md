# MilKG

将 TXT、Markdown、PDF 或 DOCX 文档转换为可预览、可下载的 SFT 问答数据集。项目提供 Vue 本地界面和命令行两种使用方式。

## 环境与依赖

| 用途 | 要求 |
| --- | --- |
| Python 后端与命令行 | Python 3.10 或更高版本 |
| WebUI 构建 | Node.js 22.18 或更高版本、npm |
| 模型推理 | 可调用的 OpenAI 兼容聊天模型 API；默认配置使用阿里云百炼 Qwen，服务器本机的 Ollama 无需 API Key |
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

Linux 服务器已经 `git clone` 项目时，在仓库目录运行：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools
./.venv/bin/python -m pip install -e '.[documents,web,neo4j]'
cd frontend
npm ci
npm run build
cd ..
./.venv/bin/python -m milkg.web
```

服务仍默认监听服务器的 `127.0.0.1:8000`。在自己的电脑上访问服务器 WebUI，可另开终端建立 SSH 端口转发：`ssh -L 8000:127.0.0.1:8000 用户名@服务器地址`，再打开本机的 `http://127.0.0.1:8000`。仅使用命令行时无需 Node.js 和前端构建。

## 使用 WebUI

启动本地服务：

```powershell
./.venv/Scripts/python -m milkg.web
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。服务默认只监听本机 `127.0.0.1`。

1. 选择“构图并生成”“只构建图谱”或“只生成 SFT”。构图任务可一次选择多个 TXT、MD、PDF、DOCX 文件，或选择整个文件夹，也可拖入文件与文件夹。每批最多 200 份、总计 200 MB，单文件上限 20 MB；不支持的文件、空文件和超限单文件会在页面提示并跳过。选中的文档在同一个任务中依次分块和抽取，合入同一张图谱，再生成一份数据集。独立生成任务无需上传文件。
2. 调用阿里云时填写 API Key，或在启动服务前设置环境变量 `ALIYUN_API_KEY`；连接服务器本机的 Ollama 时可留空。页面填写的 Key 只用于当前任务，不写入任务配置或数据集。
3. 选择本地 JSON 或 Neo4j 存储。使用 Neo4j 时填写 Bolt 地址、数据库、用户名和密码；构图选择“新建逻辑图谱”或“扩展已有图谱”。扩展或独立生成时点击“读取图谱列表”并选择目标图谱。
4. 填写模型服务地址和 API Key 后，点击“刷新模型列表”，页面会读取当前服务可用的模型，供提取与合成阶段选择；模型名也可以手动输入。阿里云百炼使用同一域名的模型列表接口，本机 Ollama 使用 `/v1/models`。设置温度和高级参数，启动任务并查看日志。
5. 生成任务完成后下载 Alpaca、ShareGPT 或 ChatML JSON。只构图任务不生成 SFT，页面会显示图谱 ID，可供下次扩展或独立生成。

任务由后端继续执行，关闭浏览器不会停止任务。重新打开页面会恢复上次查看的任务，也可在“近期任务”中切换；要主动停止，请选中运行中的任务，点击进度面板的“中断任务”。本地 Ollama 的当前请求会被关闭，后续片段和生成步骤不再执行；在线模型请求可能要等当前调用返回后才结束。已完成的片段和问答仍保留。页面显示最近 500 条事件。“下载完整日志”提供此版本之后启动的任务的全部 `process.log`。相同事件会实时打印到启动后端的终端，并写入 `output/webui/<任务 ID>/process.log`。等待模型响应超过 30 秒时会持续记录等待状态。请用上述命令启动单个后端进程；多 worker 会让运行任务分散到不同进程。若**后端进程**停止，正在运行的任务会标记为中断；重启服务后仍可查看已保存的进度和日志，符合条件的抽取任务也可从检查点继续。

如果任务在实体关系抽取阶段失败或被主动中断，且任务文件与 `extractions.json` 检查点仍在，页面会显示“从已完成片段继续抽取”。再次填写所需 API Key 或 Neo4j 密码后点击此按钮，会跳过已经完成的片段；本地模型的思考模式和最大输出 token 可以先在高级参数中调整。图谱构建或数据集生成阶段的任务暂不支持这一检查点续跑。

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

### 服务器本机 Ollama

MilKG 使用 Ollama 的 [OpenAI 兼容 `/v1/chat/completions` 接口](https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx)，并默认请求 JSON 对象格式。先在 Linux 服务器确认 Ollama 服务运行，再拉取并列出模型。下面的 `qwen3.5:4b` 和 `qwen3.5:9b` 是[官方模型库中的示例标签](https://ollama.com/library/qwen3.5/tags)；按服务器内存、显存和 `ollama list` 的实际结果替换即可。

```bash
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
ollama list
curl http://127.0.0.1:11434/api/tags
```

命令行的抽取与生成模型可以分别指定；本机 Ollama 无需设置 `MILKG_EXTRACT_KEY` 或 `MILKG_GENERATE_KEY`：

```bash
export MILKG_EXTRACT_URL='http://127.0.0.1:11434/v1'
export MILKG_EXTRACT_MODEL='qwen3.5:4b'
export MILKG_GENERATE_URL='http://127.0.0.1:11434/v1'
export MILKG_GENERATE_MODEL='qwen3.5:9b'
./.venv/bin/python -m milkg.cli build ./docs/batch1 --store neo4j --graph-action extend --graph-id YOUR_GRAPH_ID --output output/build1
./.venv/bin/python -m milkg.cli generate --store neo4j --graph-id YOUR_GRAPH_ID --output output/sft --include-atomic --format alpaca
```

WebUI 中将“服务地址”设为 `http://127.0.0.1:11434/v1`，点击“刷新模型列表”即可从本机 Ollama 读取已安装模型；也可以把 `ollama list` 中的完整名称分别填入实体关系提取和问答合成模型，API Key 留空即可。此处的 `127.0.0.1` 指运行 MilKG 后端的 Linux 服务器；即使浏览器通过 SSH 转发打开，模型请求仍由服务器发出。若只想试运行，不使用已有图谱，可在命令行执行 `run ./your_document.txt --output output/run`，或在 WebUI 选择本地 JSON 存储。Ollama 返回慢于默认在线模型时，本地接口的单次请求超时为 300 秒。WebUI 默认向本地 OpenAI 兼容接口发送 `reasoning_effort: "none"`，并将单次输出上限设为 4096 token，以免结构化抽取长时间陷入思考或重复生成；高级参数可改为使用模型默认思考模式、调整输出上限。这里的 4096 是回复上限，不是输入加回复的上下文窗口；1800 字符文档之外还要计入实体关系定义、实体清单和 JSON 回复。Ollama 默认上下文窗口为 4096 token，密集文档建议在 Ollama 服务端设置 `OLLAMA_CONTEXT_LENGTH=8192` 或 `16384` 并监控显存。若输出达到上限而 JSON 未完成，可适当提高上限或减小分块字符数。若所用模型不支持 JSON 响应格式，可在命令行加 `--no-json-mode`，但模型仍须自行输出可解析的 JSON；WebUI 默认要求 JSON 格式。

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

`graph_ref.json` 保存逻辑图谱 ID；`graph.json` 是当次快照。Neo4j 实体同时带 `MilKGEntity` 和具体实体类别标签，例如 `` `Weapon System` ``；知识关系直接使用 12 类关系名称作为 Neo4j 类型，例如 `` `Equip-Carry` ``。节点仍保留 `entity_type` 属性，关系仍保留 `relation_type` 属性。`MilKGGraph` 是图谱元数据，`MilKGDocument` 和 `MENTIONED_IN` 保存文档来源，它们不属于 14 类实体和 12 类知识关系。所有对象带 `graph_id`，可按图谱 ID 查询跨文档实体与关系。重复读取相同文档时，文档 ID 由内容哈希确定，来源不会重复累计。

旧版本保存的图谱可在 WebUI 中选择“只生成 SFT”或“扩展已有图谱”，读取图谱列表，选中目标图谱后点击“更新旧图谱的实体与关系分类”。该操作不调用模型，不重新抽取，也不改变实体、关系和文档来源数量；重复点击不会产生重复关系。也可以运行：

```powershell
./.venv/Scripts/python -m milkg.cli migrate-neo4j --store neo4j --graph-id YOUR_GRAPH_ID
```

迁移后刷新 Neo4j Browser。Browser 仅列出图中实际存在的本体类别，不一定同时出现全部 14 类实体和 12 类关系。类别名含空格、斜杠或连字符时，Cypher 查询需要使用反引号：

```cypher
MATCH (n:`Weapon System`)-[r:`Weapon-Spec`]->(s:`Technical Specification`)
RETURN n, r, s LIMIT 25
```

后续扩展旧图谱时也会自动升级其分类。

### 从 Windows 本机迁移 Neo4j 到 Linux 服务器

`migrate-neo4j` 只更新**同一个数据库内**的实体标签和关系类型；跨机器搬运图谱使用 Neo4j 的 `database dump/load`。此方式迁移整个 `neo4j` 数据库，包含所有 MilKG 逻辑图谱、原有 `graph_id`、实体、关系、文档来源、约束与索引；项目代码和 `output/` 不会随数据库转移。Neo4j 官方要求 Community 版在[离线状态 dump](https://neo4j.com/docs/operations-manual/current/backup-restore/offline-backup/)和[离线状态 load](https://neo4j.com/docs/operations-manual/current/backup-restore/restore-dump/)。先确认两端 Neo4j 版本兼容，优先使用相同版本。

1. 在 Windows 本机停止 Neo4j。若使用 Neo4j Desktop，在对应 DBMS 的安装目录 `bin` 打开终端；若使用 ZIP 安装，进入 `<NEO4J_HOME>\bin`。在 PowerShell 中执行（将目录换成自己的可写路径）：

   ```powershell
   New-Item -ItemType Directory -Force D:\neo4j-export
   .\neo4j-admin database dump neo4j --to-path=D:\neo4j-export
   Set-Location D:\neo4j-export
   scp .\neo4j.dump 用户名@服务器地址:/tmp/neo4j.dump
   ```

2. Linux 服务器上先停止 Neo4j。以下示例适用于 Debian/RPM 的 systemd 安装；若是压缩包安装，把 `neo4j-admin` 换为 `<NEO4J_HOME>/bin/neo4j-admin`。`--overwrite-destination=true` 会**替换服务器现有 `neo4j` 数据库**；服务器已有需要保留的数据时，先为它另做备份。

   ```bash
   sudo systemctl stop neo4j
   sudo install -d -o neo4j -g neo4j -m 700 /var/lib/neo4j/milkg-import
   sudo install -o neo4j -g neo4j -m 600 /tmp/neo4j.dump /var/lib/neo4j/milkg-import/neo4j.dump
   sudo -u neo4j neo4j-admin database load neo4j --from-path=/var/lib/neo4j/milkg-import --overwrite-destination=true
   sudo systemctl start neo4j
   ```

3. 在服务器上设置 Neo4j 连接环境变量并检查图谱目录；`YOUR_GRAPH_ID` 仍是 Windows 本机原来的图谱 ID，可从原有 `graph_ref.json` 或 Neo4j 的 `MilKGGraph` 节点查到。

   ```bash
   export MILKG_NEO4J_URI='bolt://127.0.0.1:7687'
   export MILKG_NEO4J_USER='neo4j'
   read -rs -p 'Neo4j password: ' MILKG_NEO4J_PASSWORD; echo
   export MILKG_NEO4J_PASSWORD
   ./.venv/bin/python -m milkg.cli graphs --store neo4j --output output/graphs
   ```

只迁移 `neo4j` 数据库时，服务器上 Neo4j 的账户和密码仍由服务器自身管理，需使用服务器原有登录凭据；[官方 dump 不包含用户和角色元数据](https://neo4j.com/docs/operations-manual/current/backup-restore/offline-backup/)。若服务器已有图谱且希望与本机图谱合并，整库 `load` 不适用，因为它会替换目标数据库。

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
