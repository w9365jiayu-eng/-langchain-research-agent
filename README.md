# LangChain Research Agent

一个用于学习和工程展示的研究助理项目，基于 LangChain 经典 ReAct Agent，实现长期向量记忆、任务规划、互联网搜索、网页抓取与 PDF 解析。

## 系统架构

```text
User
  |
  v
app.py
  |
  +-- AgentSecurityGuard    输入安全门控
  |
  +-- VectorMemory.search()  历史研究召回
  |
  +-- ResearchPlanner       生成3-5步研究计划
  |
  +-- ReAct Agent           推理与工具选择
          |
          +-- web_search
          +-- fetch_webpage
          +-- parse_pdf
  |
  +-- ReflectionEvaluator   质量评估与受控修订
  |
  +-- VectorMemory.save()    持久化研究结果
```

`app.py` 是唯一正式交互入口。`examples/` 中的脚本仅用于单独演示 Memory 和 Planning 流程。

## 项目结构

```text
langchain-research-agent/
├── app.py
├── research_agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── planner.py
│   ├── memory.py
│   ├── reflection.py
│   ├── security.py
│   ├── validation.py
│   ├── config.py
│   └── tools/
│       ├── __init__.py
│       ├── search_tool.py
│       ├── web_scraper.py
│       └── pdf_parser.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   └── eval_agent.py
├── scripts/
│   ├── __init__.py
│   └── cleanup_memory.py
├── examples/
│   ├── memory_chat.py
│   └── planning_chat.py
├── tests/
│   ├── test_agent.py
│   ├── test_memory.py
│   ├── test_reflection.py
│   ├── test_security.py
│   ├── test_security_integration.py
│   ├── test_evaluator.py
│   ├── test_integration_quality.py
│   └── data/
│       └── sample.pdf
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

运行后还会生成被 `.gitignore` 忽略的 `agent_memory/` 和 `test_agent_memory/` Chroma 数据目录。

## 环境要求

- Python 3.11+
- OpenAI-compatible LLM API
- 可访问模型和搜索服务的网络环境

## 安装

激活你已有的虚拟环境，然后执行：

```powershell
cd D:\ai_agent_project23\project03\langchain-research-agent
python -m pip install -r requirements.txt
```

## 配置

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

不要提交包含真实密钥的 `.env`。

## 运行

唯一正式入口：

```powershell
python app.py
```

每轮执行顺序：

1. 从 Chroma 检索相关历史研究。
2. Planner 根据问题和 Memory 生成研究步骤。
3. ReAct Agent 按需调用搜索、网页或 PDF 工具。
4. Reflection 对有效初稿进行质量评估与有限修订。
5. 输出通过安全检查的结构化研究报告。
6. 将问题与报告保存到长期向量记忆。

输入 `exit`、`quit`、`退出` 或按 `Ctrl+C` 可正常退出。

## 测试

```powershell
pytest
```

Agent 端到端测试需要有效的 `.env` 和网络连接。首次运行 Memory 测试时会下载 `BAAI/bge-small-zh-v1.5`。

也可以直接运行测试脚本查看完整过程：

```powershell
python tests/test_agent.py
python tests/test_memory.py
```

## 示例入口

示例脚本不是正式入口。从项目根目录运行：

```powershell
python -m examples.memory_chat
python -m examples.planning_chat
```

清理失败的历史记忆：

```powershell
python -m scripts.cleanup_memory
```

运行独立 Agent 评估：

```powershell
python -m evaluation.eval_agent
```

## 工程特性

- 保留 LangChain `create_react_agent` 与 `AgentExecutor` 经典架构。
- `verbose=False` 避免框架输出与应用输出重复。
- 最多5轮 ReAct 执行，并设置120秒时间限制。
- ChromaDB + `BAAI/bge-small-zh-v1.5` 提供持久化语义记忆。
- Planner 与 Memory 失败时安全降级，不阻断后续交互。
- 工具和网络异常均转换为可读信息，不让 CLI 直接崩溃。

## Automated Evaluation

项目通过 GitHub Actions 在以下情况自动执行测试：

- 向 `main` 分支 push。
- 创建或更新以 `main` 为目标分支的 Pull Request。

自动化流程使用 Python 3.11，并通过 Continuous Evaluation Pipeline 依次执行：

- Unit Test
- Agent Evaluation
- Security Test
- Memory Test

CI 从 GitHub Actions Secret `DEEPSEEK_API_KEY` 读取模型密钥，并映射为项目使用的 `LLM_API_KEY`。CI 禁止加载本地 `.env` 文件。请在 GitHub 仓库的 `Settings > Secrets and variables > Actions` 中配置该 Secret。

任一 pytest 或 Agent Evaluation 用例失败都会使 workflow 失败。评估完成后，`evaluation_results.json` 会作为名为 `agent-evaluation-results` 的 Artifact 上传。

本地执行完整的持续评估：

```powershell
python evaluation/run_evaluation.py
```

该入口会运行 pytest 和 Agent Evaluation，计算 Functional、Security、Efficiency 与 Overall Score，打印 Agent Health Dashboard，将完整结果写入 `evaluation_results.json`，并与 `evaluation/baseline.json` 比较。当前综合分比 baseline 低超过5分时会输出 `Performance degradation detected` 并返回非零退出码。

## Local Auto Testing

开发阶段可以使用 `pytest-watch` 获得保存即测试的快速反馈：

```text
修改 Python 代码或测试
  ↓
Ctrl + S 保存
  ↓
pytest-watch 检测文件变化
  ↓
自动执行 Continuous Evaluation Pipeline
```

直接运行：

```powershell
ptw --runner "python evaluation/run_evaluation.py"
```

也可以使用项目提供的 PowerShell 脚本。脚本会优先使用当前已激活的虚拟环境，否则尝试查找项目或工作区已有的虚拟环境：

```powershell
.\scripts\run_watch_tests.ps1
```

在 VS Code 中可以通过 `Terminal > Run Task > Watch Agent Tests` 启动相同的监听任务。使用 `Ctrl+C` 停止监听。
