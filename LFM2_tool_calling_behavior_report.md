# LFM2 Tool Calling Behavior Report

**Date:** 2025-10-25
**Model:** LFM2-Tool (1.2B parameters)
**Endpoint:** localhost:8080/v1
**Configuration:** temperature=0

## Executive Summary

LFM2 exhibits a **sharp, deterministic threshold** for tool calling based on conversation context. With 2 messages (system + user), tool calling succeeds 100% of the time. With 4 messages (adding conversation history), tool calling fails 100% of the time. This behavior is completely deterministic and predictable.

**Critical Implication:** This makes multi-step agentic workflows (tool-calling loops) extremely challenging, as the model cannot maintain tool-calling behavior once conversation history accumulates from previous tool calls.

## Background

I am building a multi-agent research system using LangGraph and evaluating LFM2 as a local alternative to larger models. The system uses tool calling for file operations via the Model Context Protocol (MCP). I noticed that after the first tool call (which succeeded), the model would respond with text explanations instead of making the second required tool call.

## Experimental Setup

I conducted comprehensive testing with multiple variables:

### Phase 1: Context Length Testing

All tests used identical explicit instructions and **verbose tool descriptions** (found to be beneficial):

**Test 1 - Minimal Context (2 messages):**
- Messages: `[SystemMessage, HumanMessage("Read file X")]`
- Tool: Verbose read_file description (~771 chars)
- Runs: 5 iterations to check consistency

**Test 2 - Medium Context (4 messages):**
- Messages: `[SystemMessage, HumanMessage, AIMessage(tool_call), ToolMessage(results)]`
- Tool: Same verbose read_file description
- Simulates: After first tool call in conversation
- Runs: 5 iterations to check consistency

**Test 3 - Full Context (4 messages, verbose):**
- Messages: Same as Test 2 but with detailed research brief (~1500 chars total)
- Tool: Same verbose read_file description
- Runs: 5 iterations to check consistency

### Phase 2: Tool Description Impact Testing

I also tested whether verbose tool schemas were causing context bloat:

**Comparison Tests:**
- Native tool (minimal description): 162 chars
- Native tool (verbose description): 771 chars
- MCP tool (verbose description): 838 chars

All tested at identical 2-message context to isolate tool description impact.

## Results

### Phase 1: Context Length Testing (5 runs each, temperature=0)

| Test | Messages | Success Rate | Consistency | Behavior |
|------|----------|-------------|------------|----------|
| 1. Minimal (2 msg) | `[System, User]` | **5/5 (100%)** | ✅ Deterministic | Always calls tool correctly |
| 2. Medium (4 msg) | `[System, User, AI, Tool]` | **0/5 (0%)** | ✅ Deterministic | Always provides text response |
| 3. Full (4 msg, verbose) | `[System, User, AI, Tool]` | **0/5 (0%)** | ✅ Deterministic | Always provides text response |

### Phase 2: Tool Description Impact (2-message context)

| Tool Type | Schema Size | Success | Key Finding |
|-----------|------------|---------|-------------|
| Native (minimal) | 162 chars | ❌ **FAIL** | Too minimal - model doesn't understand |
| Native (verbose) | 771 chars | ✅ **SUCCESS** | Verbose helps model understand tool |
| MCP (verbose) | 838 chars | ✅ **SUCCESS** | MCP performs as well as native verbose |

## Key Finding

**The same explicit instructions that work in Test 1 fail in Tests 2 and 3.** This indicates that context length, not prompt clarity, is the limiting factor.

### What I Ruled Out

1. ❌ **Insufficient instructions** - Test 1 proves LFM2 can follow the instructions when context is minimal
2. ❌ **Tool binding issues** - Same tool binding works in Test 1
3. ❌ **Fundamental inability** - The model CAN call tools, just not with longer context

### What I Confirmed

✅ **Context length sensitivity** - As context grows, LFM2 shifts from "action mode" (tool calling) to "explanation mode" (text generation), even with explicit prohibitions against text responses.

## Reproducibility

Complete test code available in: `test_context_length.py`

Run with: `python test_context_length.py`

Dependencies:
- LFM2-Tool running on localhost:8080
- LangChain MCP adapters
- Model Context Protocol filesystem server

## Implications

For agentic workflows requiring multi-step tool calling:

1. **Current limitation:** After 1-2 tool calls, conversation context becomes long enough that LFM2 defaults to text responses
2. **Workaround required:** Architecture must minimize context per decision point
3. **Potential solutions:**
   - Message filtering/pruning between tool calls
   - Separate LLM sessions for each decision
   - Stateless nodes that only receive minimal context

## Questions for Liquid AI

1. Is this expected behavior for a 1.2B parameter model?
2. Are there recommended context management strategies for multi-step tool calling?
3. Is there a context window threshold where tool calling reliability degrades?
4. Would fine-tuning or different prompting strategies improve tool calling consistency in longer contexts?

## Test Environment

- **OS:** macOS (Darwin 24.5.0)
- **Python:** 3.13
- **LangChain:** Latest (2025)
- **Model Format:** GGUF via llama.cpp
- **Inference Server:** llama-server (OpenAI-compatible API)

---

*This report documents observed behavior during development of a deep research system. I appreciate any insights into whether this is expected behavior or if there are optimization opportunities.*
