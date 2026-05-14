# MathRAG

基于 **FastAPI + FAISS + Embedding API + DeepSeek(OpenAI 兼容) API** 的数学问答 RAG 原型系统。

该项目面向“数学助教/教学演示”场景：
- 先从结构化知识库中召回相关知识；
- 再由大模型生成结构化回答（答案、步骤、参考知识、追问建议）；
- 同时提供可直接访问的浏览器前端页面与 API。

---

## 1. 核心能力

- 数学问答（RAG 检索增强）
- FAISS 向量检索（支持内积检索）
- 结构化回答输出（`answer` / `steps` / `references` / `related_questions`）
- 简单多轮对话历史输入（`history`）
- FastAPI 后端 + 原生前端静态页面
- Docker / Docker Compose 部署支持
- 基于 `pytest` 的 API 测试样例

---

## 2. 项目结构

```text
MathRAG/
├─ app/
│  ├─ api/                # 路由层
│  ├─ core/               # 配置与日志
│  ├─ frontend/           # 前端静态页面
│  ├─ schemas/            # 请求/响应模型
│  ├─ services/           # embedding/retriever/llm/rag 主逻辑
│  └─ utils/              # 文本清洗、提示词构建、后处理
├─ data/
│  ├─ raw/                # 原始知识库
│  ├─ processed/          # chunk 化后的知识数据
│  └─ index/              # FAISS 索引与映射
├─ scripts/               # 构建知识库、构建索引、检索与RAG调试脚本
├─ tests/                 # API 测试
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ run.py
└─ README.md
```

---

## 3. 环境要求

推荐：
- Python 3.11
- Linux / macOS / Windows
- 可用的 Embedding API Key
- 可用的 DeepSeek API Key（OpenAI 兼容接口）

> 说明：项目中包含 `faiss-cpu`，在不同平台下安装可能稍有差异。优先使用 Python 3.11 + 虚拟环境。

---

## 4. 安装

### 4.1 本地安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5. 配置 `.env`

在项目根目录创建 `.env` 文件：

```env
# App
APP_NAME=MathRAG MVP
APP_HOST=127.0.0.1
APP_PORT=8000
DEBUG=true

# Embedding
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
EMBEDDING_BATCH_SIZE=10
EMBEDDING_TIMEOUT=60
EMBEDDING_NORMALIZE=true

# LLM (DeepSeek OpenAI-Compatible)
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-reasoner
LLM_TIMEOUT=120
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.2
LLM_RETURN_REASONING=false

# Retrieval
TOP_K=3
USE_INNER_PRODUCT=true
```

---

## 6. 数据准备与索引构建

### 6.1 原始知识数据位置

```text
data/raw/math_knowledge_seed.jsonl
```

### 6.2 构建知识 chunk

```bash
python -m scripts.build_kb
```

### 6.3 构建向量索引

```bash
python -m scripts.build_index
```

成功后会生成：

```text
data/processed/kb_chunks.jsonl
data/index/faiss.index
data/index/id_map.json
```

### 6.4 从公开数学网站导入知识数据

项目提供了批量导入脚本，用于从公开数学知识来源检索页面、清洗文本、切分 chunk、调用大语言模型整理为严格 JSON，并追加保存到原始知识库文件：

```text
data/raw/math_knowledge_seed.jsonl
```

当前优先支持的数据源：

- `wikipedia`：通过 MediaWiki API 获取数学条目。
- `wikibooks`：通过 MediaWiki API 获取数学教材章节。
- `proofwiki`：通过 MediaWiki API 获取定义、定理、引理、证明等内容；部分网络环境可能返回 403。
- `planetmath`：受限 HTML 抓取，稳定性取决于站点访问情况。

暂不将 MathWorld、OpenStax、arXiv、Math StackExchange 作为主要批量来源，因为这些来源存在许可、API、数据使用限制或质量筛选问题，需要单独设计导入策略。

示例：从 Wikipedia 和 Wikibooks 导入 `derivative` 相关内容：

```powershell
.\.venv\Scripts\python.exe -m scripts.import_math_knowledge `
  --sources wikipedia wikibooks `
  --keywords derivative `
  --limit-per-source 2 `
  --stage undergraduate `
  --course "Calculus" `
  --category calculus
```

Linux / macOS 可写为：

```bash
python -m scripts.import_math_knowledge \
  --sources wikipedia wikibooks \
  --keywords derivative \
  --limit-per-source 2 \
  --stage undergraduate \
  --course "Calculus" \
  --category calculus
```

常用参数：

- `--sources`：数据源列表，可选 `proofwiki`、`planetmath`、`wikibooks`、`wikipedia`。
- `--keywords`：搜索关键词，可传多个。
- `--limit-per-source`：每个数据源、每个关键词最多取多少条搜索结果，默认 `3`。
- `--max-chunk-chars`：每个 LLM chunk 的最大字符数，默认 `6000`。
- `--delay-seconds`：页面请求间隔，默认 `1.0` 秒。
- `--stage`：可选学段，取值为 `primary`、`junior_secondary`、`senior_secondary`、`undergraduate`。
- `--course`：课程名提示，例如 `"Calculus"`。
- `--category`：知识分类提示，例如 `calculus`。
- `--output`：合格知识点输出文件，默认 `data/raw/math_knowledge_seed.jsonl`。
- `--error-output`：不合格数据或抓取错误输出文件，默认 `data/raw/math_knowledge_import_errors.jsonl`。

脚本写入 `math_knowledge_seed.jsonl` 时会严格保持原知识库 JSONL 格式，每行只包含：

```text
id, category, stage, course, title, keywords, content, example, steps, prerequisites, difficulty
```

不会把接口响应包装字段（如 `records`、`saved_count`、`next_steps`）写入知识库。

导入后建议先校验原始知识库：

```bash
python -m scripts.validate_seed_jsonl
```

如果校验通过，再重新构建 chunk 和向量索引：

```bash
python -m scripts.build_kb
python -m scripts.build_index
```

### 6.5 从本地 PDF 数据湖抽取文本

如果不希望从网页抓取，可以把 PDF 文件放到：

```text
data/data_lake/
```

然后先抽取、清洗为文本 chunk 集：

```powershell
.\.venv\Scripts\python.exe -m scripts.import_pdf_knowledge `
  --data-dir data\data_lake `
  --text-output data\processed\pdf_text_chunks.jsonl `
  --max-chunk-chars 3000
```

默认只生成清洗后的文本集，不调用大语言模型，也不会写入 `math_knowledge_seed.jsonl`。文本集字段包括来源 PDF 路径、PDF 标题、chunk 序号、清洗后的文本和文本长度。

如果确认文本质量可以接受，再加 `--import-to-knowledge`，让模型把 PDF 文本 chunk 整理为中文知识点并追加到原始知识库：

```powershell
.\.venv\Scripts\python.exe -m scripts.import_pdf_knowledge `
  --data-dir data\data_lake `
  --max-chunks 20 `
  --max-chunk-chars 3000 `
  --import-to-knowledge `
  --stage senior_secondary `
  --course "高中数学" `
  --category "高中数学"
```

常用参数：

- `--data-dir`：PDF 数据湖目录，默认 `data/data_lake`。
- `--text-output`：清洗后的 PDF 文本 chunk JSONL，默认 `data/processed/pdf_text_chunks.jsonl`。
- `--max-chunk-chars`：每个文本 chunk 的最大字符数，默认 `4000`。
- `--max-chunks`：最多处理多少个文本 chunk，适合先小批量试跑。
- `--import-to-knowledge`：启用 LLM 结构化写入知识库；不加时只抽文本。
- `--stage` / `--course` / `--category`：写入知识库时给模型的学段、课程和分类提示。
- `--error-output`：PDF 知识点导入错误文件，默认 `data/raw/pdf_knowledge_import_errors.jsonl`。

PDF 文本抽取依赖 `pypdf`。如果提示缺少依赖，请重新安装：

```bash
pip install -r requirements.txt
```

---

## 7. 调试脚本

### 7.1 仅检索验证

```bash
python -m scripts.demo_query --question "x^2+4x+3=0 怎么解？" --show-context
```

交互模式：

```bash
python -m scripts.demo_query --interactive --show-context
```

### 7.2 RAG 端到端验证

```bash
python -m scripts.test_rag --question "x^2+4x+3=0 怎么解？" --show-references
```

---

## 8. 启动服务

### 8.1 本地启动

```bash
python run.py
```

或开发模式：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

启动后访问：
- 首页：`http://127.0.0.1:8000/`
- Swagger：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

### 8.2 Docker Compose 启动

```bash
docker compose up -d --build
```

查看日志：

```bash
docker compose logs -f mathrag
```

停止：

```bash
docker compose down
```

---

## 9. API 示例

### 9.1 `POST /api/chat`

请求：

```json
{
  "question": "x^2+4x+3=0 怎么解？",
  "history": [
    {"role": "user", "content": "我不会解一元二次方程"}
  ],
  "top_k": 3
}
```

响应（示例）：

```json
{
  "question": "x^2+4x+3=0 怎么解？",
  "answer": "可因式分解得到 x=-1 或 x=-3。",
  "steps": [
    "将方程整理为标准形式。",
    "因式分解为 (x+1)(x+3)=0。",
    "分别令因式为0得到两个根。"
  ],
  "used_knowledge": ["因式分解法解一元二次方程"],
  "related_questions": ["如何用求根公式解？", "什么情况下适合因式分解？"],
  "references": [
    {
      "rank": 1,
      "score": 0.91,
      "index": 12,
      "chunk_id": "k0001_chunk_0",
      "source_id": "k0001",
      "category": "quadratic_equation",
      "stage": "junior_secondary",
      "course": "初中代数",
      "title": "因式分解法解一元二次方程",
      "keywords": ["一元二次方程", "因式分解"],
      "content": "...",
      "example": "...",
      "steps": ["..."],
      "prerequisites": ["整式乘法"],
      "difficulty": "easy",
      "answer_context": "...",
      "retrieval_text": "...",
      "source_line": 1,
      "metadata": {}
    }
  ],
  "reasoning_content": null
}
```

---

## 10. 测试

运行测试：

```bash
pytest -q
```

当前测试主要覆盖：
- `/api/chat` 成功响应结构
- `history` 参数透传
- 参数校验（空问题、非法 `top_k`）
- 管道异常时的 HTTP 状态码与错误信息

---

## 11. 常见问题

### 11.1 `ModuleNotFoundError: No module named 'app'`

请确保在项目根目录下执行，并优先使用模块方式：

```bash
python -m scripts.build_index
```

### 11.2 大模型接口报鉴权/余额错误

- 检查 `LLM_API_KEY` 是否正确；
- 检查 DeepSeek 账户余额与调用权限；
- 检查 `LLM_BASE_URL` 是否可访问。

### 11.3 首页能打开但样式丢失

确认以下静态文件存在：

```text
app/frontend/index.html
app/frontend/style.css
app/frontend/app.js
```

---

## 12. 后续可扩展方向

- 引入 rerank 提升召回精度
- 增加公式渲染（如 KaTeX）
- 增强多轮上下文管理与记忆策略
- 增加评测集与自动化评估脚本
- 扩展更多学段与题型知识库

---

## License

本项目主要用于教学、演示与研究原型。
