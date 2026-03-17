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