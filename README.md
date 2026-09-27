# MilKG

将 TXT、Markdown、PDF 或 DOCX 文档转换为可预览、可下载的 SFT 问答数据集。项目提供 Vue 本地界面和命令行两种使用方式。

## 环境与依赖

| 用途 | 要求 |
| --- | --- |
| Python 后端与命令行 | Python 3.10 或更高版本 |
| WebUI 构建 | Node.js 22.18 或更高版本、npm |
| 在线生成 | 可调用的 OpenAI 兼容聊天模型 API 与 API Key；默认配置使用阿里云百炼 Qwen |

Python 依赖由 `pyproject.toml` 管理：

- 核心：`networkx`。
- 读取 PDF / DOCX：`pypdf`、`python-docx`，对应 `documents` 可选依赖。
- Web 服务：`fastapi`、`uvicorn`、`python-multipart`，对应 `web` 可选依赖。
- 测试：`pytest`，对应 `test` 可选依赖。

前端依赖由 `frontend/package-lock.json` 锁定，主要是 Vue 3、Vite 和 Vue 的 Vite 插件。只使用命令行时无需安装 Node.js。

## 安装

以下命令适用于 Windows PowerShell：

```powershell
git clone https://github.com/wangwangquan98/MilKG.git
cd MilKG
python -m venv .venv
./.venv/Scripts/python -m pip install --upgrade pip
./.venv/Scripts/python -m pip install -e ".[documents,web]"
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

1. 上传 TXT、MD、PDF 或 DOCX 文件，单文件上限为 20 MB。
2. 填写 API Key，或在启动服务前设置环境变量 `ALIYUN_API_KEY`。页面填写的 Key 只用于当前任务，不写入任务配置或数据集。
3. 选择实体关系提取模型、问答生成模型、API 端点和温度；需要时展开“高级参数”设置分块、题型与生成数量。
4. 点击“开始生成数据集”，查看当前阶段、进度、日志和已通过校验的问答。
5. 完成后选择 Alpaca、ShareGPT 或 ChatML 格式下载 JSON。

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
