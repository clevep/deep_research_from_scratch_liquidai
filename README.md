# 🧱 Deep Research From Scratch 

Deep research has broken out as one of the most popular agent applications. [OpenAI](https://openai.com/index/introducing-deep-research/), [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system), [Perplexity](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research), and [Google](https://gemini.google/overview/deep-research/?hl=en) all have deep research products that produce comprehensive reports using [various sources](https://www.anthropic.com/news/research) of context. There are also many [open](https://huggingface.co/blog/open-deep-research) [source](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) implementations. We built an [open deep researcher](https://github.com/langchain-ai/open_deep_research) that is simple and configurable, allowing users to bring their own models, search tools, and MCP servers. In this repo, we'll build a deep researcher from scratch! Here is a map of the major pieces that we will build:

![overview](https://github.com/user-attachments/assets/b71727bd-0094-40c4-af5e-87cdb02123b4)

## 🚀 Quickstart 

### Prerequisites

- **Node.js and npx** (required for MCP server in notebook 3):
```bash
# Install Node.js (includes npx)
# On macOS with Homebrew:
brew install node

# On Ubuntu/Debian:
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation:
node --version
npx --version
```

- Ensure you're using Python 3.11 or later.
- This version is required for optimal compatibility with LangGraph.
```bash
python3 --version
```
- [uv](https://docs.astral.sh/uv/) package manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Update PATH to use the new uv version
export PATH="/Users/$USER/.local/bin:$PATH"
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/langchain-ai/deep_research_from_scratch
cd deep_research_from_scratch
```

2. Install the package and dependencies (this automatically creates and manages the virtual environment):
```bash
uv sync
```

3. Create a `.env` file in the project root with your API keys:
```bash
# Create .env file
touch .env
```

Add your API keys to the `.env` file:
```env
# Required for research agents with external search
TAVILY_API_KEY=your_tavily_api_key_here

# Required for model usage
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: For evaluation and tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep_research_from_scratch
```

4. Run notebooks or code using uv:
```bash
# Run Jupyter notebooks directly
uv run jupyter notebook

# Or activate the virtual environment if preferred
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
jupyter notebook
```

## Background 

Research is an open‑ended task; the best strategy to answer a user request can’t be easily known in advance. Requests can require different research strategies and varying levels of search depth. Consider this request. 

[Agents](https://langchain-ai.github.io/langgraph/tutorials/workflows/#agent) are well suited to research because they can flexibly apply different strategies, using intermediate results to guide their exploration. Open deep research uses an agent to conduct research as part of a three step process:

1. **Scope** – clarify research scope
2. **Research** – perform research
3. **Write** – produce the final report

## 📝 Organization 

This repo contains 5 tutorial notebooks that build a deep research system from scratch:

### 📚 Tutorial Notebooks

#### 1. User Clarification and Brief Generation (`notebooks/1_scoping.ipynb`)
**Purpose**: Clarify research scope and transform user input into structured research briefs

**Key Concepts**:
- **User Clarification**: Determines if additional context is needed from the user using structured output
- **Brief Generation**: Transforms conversations into detailed research questions
- **LangGraph Commands**: Using Command system for flow control and state updates
- **Structured Output**: Pydantic schemas for reliable decision making

**Implementation Highlights**:
- Two-step workflow: clarification → brief generation
- Structured output models (`ClarifyWithUser`, `ResearchQuestion`) to prevent hallucination
- Conditional routing based on clarification needs
- Date-aware prompts for context-sensitive research

**What You'll Learn**: State management, structured output patterns, conditional routing

---

#### 2. Research Agent with Custom Tools (`notebooks/2_research_agent.ipynb`)
**Purpose**: Build an iterative research agent using external search tools

**Key Concepts**:
- **Agent Architecture**: LLM decision node + tool execution node pattern
- **Sequential Tool Execution**: Reliable synchronous tool execution
- **Search Integration**: Tavily search with content summarization
- **Tool Execution**: ReAct-style agent loop with tool calling

**Implementation Highlights**:
- Synchronous tool execution for reliability and simplicity
- Content summarization to compress search results
- Iterative research loop with conditional routing
- Rich prompt engineering for comprehensive research

**What You'll Learn**: Agent patterns, tool integration, search optimization, research workflow design

---

#### 3. Research Agent with MCP (`notebooks/3_research_agent_mcp.ipynb`)
**Purpose**: Integrate Model Context Protocol (MCP) servers as research tools

**Key Concepts**:
- **Model Context Protocol**: Standardized protocol for AI tool access
- **MCP Architecture**: Client-server communication via stdio/HTTP
- **LangChain MCP Adapters**: Seamless integration of MCP servers as LangChain tools
- **Local vs Remote MCP**: Understanding transport mechanisms

**Implementation Highlights**:
- `MultiServerMCPClient` for managing MCP servers
- Configuration-driven server setup (filesystem example)
- Rich formatting for tool output display
- Async tool execution required by MCP protocol (no nested event loops needed)

**What You'll Learn**: MCP integration, client-server architecture, protocol-based tool access

---

#### 4. Research Supervisor (`notebooks/4_research_supervisor.ipynb`)
**Purpose**: Multi-agent coordination for complex research tasks

**Key Concepts**:
- **Supervisor Pattern**: Coordination agent + worker agents
- **Parallel Research**: Concurrent research agents for independent topics using parallel tool calls
- **Research Delegation**: Structured tools for task assignment
- **Context Isolation**: Separate context windows for different research topics

**Implementation Highlights**:
- Two-node supervisor pattern (`supervisor` + `supervisor_tools`)
- Parallel research execution using `asyncio.gather()` for true concurrency
- Structured tools (`ConductResearch`, `ResearchComplete`) for delegation
- Enhanced prompts with parallel research instructions
- Comprehensive documentation of research aggregation patterns

**What You'll Learn**: Multi-agent patterns, parallel processing, research coordination, async orchestration

---

#### 5. Full Multi-Agent Research System (`notebooks/5_full_agent.ipynb`)
**Purpose**: Complete end-to-end research system integrating all components

**Key Concepts**:
- **Three-Phase Architecture**: Scope → Research → Write
- **System Integration**: Combining scoping, multi-agent research, and report generation
- **State Management**: Complex state flow across subgraphs
- **End-to-End Workflow**: From user input to final research report

**Implementation Highlights**:
- Complete workflow integration with proper state transitions
- Supervisor and researcher subgraphs with output schemas
- Final report generation with research synthesis
- Thread-based conversation management for clarification

**What You'll Learn**: System architecture, subgraph composition, end-to-end workflows

---

### 🎯 Key Learning Outcomes

- **Structured Output**: Using Pydantic schemas for reliable AI decision making
- **Async Orchestration**: Strategic use of async patterns for parallel coordination vs synchronous simplicity
- **Agent Patterns**: ReAct loops, supervisor patterns, multi-agent coordination
- **Search Integration**: External APIs, MCP servers, content processing
- **Workflow Design**: LangGraph patterns for complex multi-step processes
- **State Management**: Complex state flows across subgraphs and nodes
- **Protocol Integration**: MCP servers and tool ecosystems

Each notebook builds on the previous concepts, culminating in a production-ready deep research system that can handle complex, multi-faceted research queries with intelligent scoping and coordinated execution. 



==========


# Install & Run the llama.cpp C++ Server (OpenAI-compatible)

This guide shows how to build and run the **C++ HTTP server** from the `llama.cpp` project.  
It exposes OpenAI-style endpoints (`/v1/chat/completions`) and works great with LangChain’s `init_chat_model`.

---

## 1) Prerequisites

### macOS (Apple Silicon or Intel)
- Command Line Tools:  
  ```bash
  xcode-select --install
  ```
- Homebrew packages:  
  ```bash
  brew install cmake
  ```
- Model file in **GGUF** format (e.g. `LFM2-1.2B-Tool-Q4_K_M.gguf`)

### Linux
- Build tools:  
  ```bash
  sudo apt-get update && sudo apt-get install -y build-essential cmake git
  ```
- (Optional) CUDA Toolkit installed if you want GPU acceleration

---

## 2) Get the Source

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

> The C++ server binary is built via **CMake** and is typically called `llama-server` (sometimes just `server`) under `build/bin/`.

---

## 3) Build

### macOS + Metal (recommended on Apple Silicon)
```bash
cmake -S . -B build -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Linux (CPU only)
```bash
cmake -S . -B build -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Linux (CUDA GPU)
```bash
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Verify the binary exists:**
```bash
ls -l build/bin
# expect: ./build/bin/llama-server  (or ./build/bin/server)
```

---

## 4) Run the Server

Basic, stable startup (single request at a time, no continuation batching):

```bash
./build/bin/llama-server \
  -m /absolute/path/to/your-model.gguf \
  -c 32768 \
  --port 8080 \
  --parallel 1 \
  --no-cont-batching
```

**Flags explained**
- `-m` / `--model` — path to the GGUF model
- `-c` / `--ctx-size` — context window (32768 for LFM2 models, which support up to 32,768 tokens)
- `--port` — HTTP port
- `--parallel` — number of parallel sequences
- `--no-cont-batching` — disables continuation batching (avoids KV cache reuse edge cases)

> 💡 On macOS/Metal: the first run compiles Metal kernels in memory; you’ll see initialization messages. That’s normal.

---

## 5) Quick Validation

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"lfm2",
    "messages":[{"role":"user","content":"Say hi in one short sentence."}],
    "max_tokens":32,
    "cache_prompt": false,
    "stream": false
  }'
```

You should get a small JSON response with an assistant message.

---

## 6) Use with LangChain (`init_chat_model`)

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

chat = init_chat_model(
    model="lfm2",                               # arbitrary label
    model_provider="openai",                    # OpenAI-compatible
    base_url="http://127.0.0.1:8080/v1",        # C++ server endpoint
    api_key="sk-no-key",                        # any non-empty string
    temperature=0.2,
).bind(
    response_format={"type": "text"},           # avoid JSON grammar on router steps
    max_tokens=256,
    extra_body={"cache_prompt": False, "stream": False}
)

print(chat.invoke([HumanMessage(content="Say hi in one short sentence.")]).content)
```

---

## 7) Common Troubleshooting

### “invalid argument: 0” after `--cont-batching`
Use the boolean flag form:
- Disable: `--no-cont-batching` (or `-nocb`)
- Enable: `--cont-batching` (no value)

### 500 error: `tool_choice param requires --jinja`
The C++ server expects **Jinja chat templates** if you send `tool_choice`/`tools`.  
**Fix:** Don’t send `tool_choice` (or don’t bind tools). Prefer **client-side tools** (have the model return JSON args; you parse and call the function).  
If you *must* do server-side tools:
- start server with `--jinja --chat-template <template.jinja>`,
- then include `tools`/`tool_choice` in requests.

### KV cache / decode errors
- Keep `--no-cont-batching` for stability.  
- Keep prompts short on routing/clarify nodes; set `extra_body={"cache_prompt": false}` in requests.  
- Avoid complex JSON grammars unless needed.

### Port already in use
Pick another port:
```bash
--port 8081
```

### Binary name confusion
Some builds produce `build/bin/llama-server`, others `build/bin/server`.  
Use whichever exists in `build/bin`.

---

## 8) (Optional) Run as a Background Service

### macOS (launchctl example)
Create `~/Library/LaunchAgents/com.llama.server.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.llama.server</string>
  <key>ProgramArguments</key>
  <array>
    <string>/absolute/path/llama.cpp/build/bin/llama-server</string>
    <string>-m</string><string>/absolute/path/model.gguf</string>
    <string>-c</string><string>32768</string>
    <string>--port</string><string>8080</string>
    <string>--parallel</string><string>1</string>
    <string>--no-cont-batching</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/llama-server.out</string>
  <key>StandardErrorPath</key><string>/tmp/llama-server.err</string>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.llama.server.plist
launchctl start com.llama.server
```

### Linux (systemd example)
`/etc/systemd/system/llama-server.service`:
```ini
[Unit]
Description=llama.cpp HTTP Server
After=network.target

[Service]
ExecStart=/opt/llama.cpp/build/bin/llama-server -m /opt/models/model.gguf -c 32768 --port 8080 --parallel 1 --no-cont-batching
Restart=always
User=llama
WorkingDirectory=/opt/llama.cpp
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
```

---

## 9) Performance Tips

- Increase `--parallel` gradually (2, 3, …) if you need concurrency.
- On Apple Silicon, Metal is fast out of the box; no extra flags needed beyond `-DGGML_METAL=ON`.
- For LFM2 models, use `-c 32768` to leverage the full context window for research tasks with long webpages and documents.
- For larger models, watch memory. The context size impacts memory usage.
- Prefer quantized models (Q4_K_M / Q4_K_S / Q5_K_M) for speed and memory savings.

---

## 10) Quick Checklist

- [ ] Built with `-DLLAMA_BUILD_SERVER=ON`  
- [ ] Can run `./build/bin/llama-server --help`  
- [ ] Starts with `--parallel 1 --no-cont-batching`  
- [ ] `curl` test returns a reply  
- [ ] LangChain points to `http://127.0.0.1:8080/v1`

---

If you’d like, you can create a simple `build_and_run.sh` script to rebuild and launch the server automatically for your setup.


======

# README — Download & Use Liquid AI LFM2 Models

This guide explains how to download and use **two Liquid AI LFM2 models** for the deep research system. We use a dual-model setup for optimal performance:

- **LFM2-1.2B-Tool** (port 8080): For tool-calling research agent
- **LFM2-1.2B** (port 8081): For text generation (compression & summarization)

---

## 1) Model Overview

### LFM2-1.2B-Tool (Research Agent)
- **Model:** [`LiquidAI/LFM2-1.2B-Tool-GGUF`](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool-GGUF)
- **Purpose:** Main research agent with tool calling capabilities
- **Size:** ~1.2B parameters (697MB Q4_K_M)
- **Architecture:** Fine-tuned for **function/tool-use**
- **Port:** 8080

### LFM2-1.2B (Base Model)
- **Model:** [`LiquidAI/LFM2-1.2B-GGUF`](https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF)
- **Purpose:** Research compression & webpage summarization
- **Size:** ~1.2B parameters (697MB Q4_K_M)
- **Architecture:** Base model optimized for plain text generation
- **Port:** 8081

**Why two models?** The Tool model is specialized for function calling, which can cause it to output tool call JSON instead of narrative text. The base model excels at plain text generation for compression tasks.

---

## 2) Download Both Models

Run the following in your project models directory:

```bash
cd ./models

# Download LFM2-Tool model (for research agent)
curl -L -o LFM2-1.2B-Tool-Q4_K_M.gguf \
  https://huggingface.co/LiquidAI/LFM2-1.2B-Tool-GGUF/resolve/main/LFM2-1.2B-Tool-Q4_K_M.gguf

# Download LFM2 base model (for compression)
curl -L -o LFM2-1.2B-Q4_K_M.gguf \
  https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/resolve/main/LFM2-1.2B-Q4_K_M.gguf
```

> ✅ Tip:
> You can also use `huggingface-cli` if installed:
> ```bash
> huggingface-cli download LiquidAI/LFM2-1.2B-Tool-GGUF LFM2-1.2B-Tool-Q4_K_M.gguf --local-dir ./models
> huggingface-cli download LiquidAI/LFM2-1.2B-GGUF LFM2-1.2B-Q4_K_M.gguf --local-dir ./models
> ```

Verify both models are downloaded:
```bash
ls -lh ./models/
# Should show both:
# LFM2-1.2B-Tool-Q4_K_M.gguf (~697MB)
# LFM2-1.2B-Q4_K_M.gguf (~697MB)
```

---

## 3) Run Both Servers

Assuming you've already built `llama-server` (see [llama.cpp build guide](https://github.com/ggml-org/llama.cpp)), start **two separate server instances**:

### Terminal 1 - Tool Model (Port 8080)

```bash
./llama.cpp/build/bin/llama-server \
  -m ./models/LFM2-1.2B-Tool-Q4_K_M.gguf \
  -c 32768 \
  --port 8080 \
  --parallel 1 \
  --no-cont-batching
```

### Terminal 2 - Base Model (Port 8081)

```bash
./llama.cpp/build/bin/llama-server \
  -m ./models/LFM2-1.2B-Q4_K_M.gguf \
  -c 32768 \
  --port 8081 \
  --parallel 1 \
  --no-cont-batching
```

> **Note:** LFM2 models support up to 32,768 tokens context window. Using `-c 32768` ensures you can process longer webpages and research content without truncation errors.

You should see both servers start:
```
Terminal 1: HTTP server listening at http://127.0.0.1:8080
Terminal 2: HTTP server listening at http://127.0.0.1:8081
```

---

## 4) Test Both Servers

### Test Tool Model (Port 8080)
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lfm2",
    "messages": [{"role":"user","content":"Say hi in one short sentence."}],
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

### Test Base Model (Port 8081)
```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lfm2",
    "messages": [{"role":"user","content":"Say hi in one short sentence."}],
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

Both should return:
```json
{
  "choices": [{
    "message": {"role": "assistant", "content": "Hello there!"}
  }]
}
```

---

## 5) Use from LangChain

The deep research notebooks are configured to use both servers:

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# Tool model (port 8080) - for research agent with tool calling
research_model = init_chat_model(
    model="lfm2",
    model_provider="openai",
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-no-key",
    temperature=0.2,
)

# Base model (port 8081) - for compression and summarization
compression_model = init_chat_model(
    model="lfm2",
    model_provider="openai",
    base_url="http://127.0.0.1:8081/v1",
    api_key="sk-no-key",
    temperature=0.2,
    max_tokens=32000,
)

print(compression_model.invoke([HumanMessage(content="Summarize the purpose of Liquid AI.")]).content)
```

---

## 6) Memory Requirements

**Running both models simultaneously:**
- Model weights: ~1.4 GB total (697MB each)
- KV cache (32k context): ~4-6 GB total
- **Total: ~6-8 GB RAM**

**Recommended specs:**
- ✅ 16GB+ RAM: Comfortable
- ✅ Apple Silicon (M2/M3): Excellent with unified memory and Metal acceleration
- ⚠️ 8GB RAM: Possible but tight

---

## 7) Notes

- `LFM2-1.2B-Tool` supports **function/tool call generation** for the research agent
- `LFM2-1.2B` (base) provides **clean plain text generation** for compression tasks
- Both work excellently on Apple Silicon via Metal or CPU-only Linux
- For stability, use:
  - `--parallel 1`
  - `--no-cont-batching`
  - `-c 32768` for full context window

---

## 8) References

- Tool model: [LiquidAI/LFM2-1.2B-Tool-GGUF on Hugging Face](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool-GGUF)
- Base model: [LiquidAI/LFM2-1.2B-GGUF on Hugging Face](https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF)
- llama.cpp project: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)