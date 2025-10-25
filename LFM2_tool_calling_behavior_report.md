# LFM2 Tool Calling Behavior Report

**Date:** 2025-10-25
**Model:** LFM2-Tool (1.2B parameters)
**Endpoint:** localhost:8080/v1
**Configuration:** temperature=0

## Summary

We observed that LFM2's ability to make tool calls degrades significantly as conversation context length increases, even when given explicit instructions to call tools. With minimal context, the model successfully calls tools, but with moderate conversation history (same explicit instructions), it switches to text responses instead of tool calls.

## Background

We are building a multi-agent research system using LangGraph and evaluating LFM2 as a local alternative to larger models. The system uses tool calling for file operations via the Model Context Protocol (MCP). We noticed that after the first tool call (which succeeded), the model would respond with text explanations instead of making the second required tool call.

## Experimental Setup

We tested LFM2 with three different context lengths, all with identical explicit instructions prohibiting text responses:

### System Prompt (all tests)
```
CRITICAL: Your response MUST be ONLY a tool call. DO NOT write text.
DO NOT say "I will read the file..."
DO NOT explain what you're doing.
ONLY call the read_file tool with the file path.
```

### Test Conditions

**Test 1 - Minimal Context (~150 chars):**
- Messages: `[SystemMessage, HumanMessage("Read file X")]`
- Total: 2 messages

**Test 2 - Medium Context (~400 chars):**
- Messages: `[SystemMessage, HumanMessage("Find coffee shops"), AIMessage(tool_call), ToolMessage(results)]`
- Total: 4 messages (simulating after first tool call)

**Test 3 - Full Context (~1300 chars):**
- Messages: Same as Test 2 but with detailed research brief and longer system prompt
- Total: 4 messages with verbose content

## Results

| Test | Context Size | Tool Call Success | Observed Behavior |
|------|-------------|-------------------|-------------------|
| 1. Minimal | ~150 chars | ✅ **SUCCESS** | Correctly called `read_file` tool |
| 2. Medium | ~400 chars | ❌ **FAIL** | Text response: "Here is the content of the file..." |
| 3. Full | ~1300 chars | ❌ **FAIL** | Text response: "The directory contains..." |

## Key Finding

**The same explicit instructions that work in Test 1 fail in Tests 2 and 3.** This indicates that context length, not prompt clarity, is the limiting factor.

### What We Ruled Out

1. ❌ **Insufficient instructions** - Test 1 proves LFM2 can follow the instructions when context is minimal
2. ❌ **Tool binding issues** - Same tool binding works in Test 1
3. ❌ **Fundamental inability** - The model CAN call tools, just not with longer context

### What We Confirmed

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

*This report documents observed behavior during development of a deep research system. We appreciate any insights into whether this is expected behavior or if there are optimization opportunities.*
