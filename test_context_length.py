"""Test script to determine if LFM2's issue is context length or behavioral tendency.

This script tests the model's behavior with different context lengths to isolate
whether the problem is:
A) Context length overwhelming the model
B) Behavioral tendency to explain rather than act
"""

import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from pathlib import Path

# Initialize LFM2-Tool model (same config as research agent)
model = init_chat_model(
    model="lfm2",
    model_provider="openai",
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key",
    temperature=0,
)

# Get current directory
current_dir = Path(__file__).parent

# MCP client configuration
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(current_dir / "src/deep_research_from_scratch/files")
        ],
        "transport": "stdio"
    }
}

async def run_tests():
    """Run controlled experiments with different context lengths."""

    # Initialize MCP client and get tools
    client = MultiServerMCPClient(mcp_config)
    mcp_tools = await client.get_tools()

    # Get read_file tool
    read_file_tool = next((t for t in mcp_tools if t.name in ["read_file", "read_text_file"]), None)
    if not read_file_tool:
        print("❌ read_file tool not found!")
        return

    # Bind tools to model
    model_with_tools = model.bind_tools([read_file_tool])

    print("="*80)
    print("CONTEXT LENGTH EXPERIMENT")
    print("="*80)
    print()

    # ========================================
    # TEST 1: Minimal Context with explicit instructions
    # ========================================
    print("📝 TEST 1: MINIMAL CONTEXT with explicit tool calling instructions")
    print("-" * 80)
    print("Hypothesis: Explicit instructions overcome behavioral tendency even with minimal context")
    print()

    test1_messages = [
        SystemMessage(content="""You have a read_file tool.

CRITICAL: Your response MUST be ONLY a tool call. DO NOT write text.
DO NOT say "I will read the file..."
DO NOT explain what you're doing.
ONLY call the read_file tool with the file path."""),
        HumanMessage(content="Read the file at /Users/cleve/dev/deep_research_from_scratch_liquidai/src/deep_research_from_scratch/files/coffee_shops_sf.md")
    ]

    print("Input messages:")
    for i, msg in enumerate(test1_messages, 1):
        print(f"  {i}. {msg.__class__.__name__}: {msg.content[:100]}...")
    print()

    response1 = model_with_tools.invoke(test1_messages)

    print("Response:")
    if response1.tool_calls:
        print(f"  ✅ Called tool: {response1.tool_calls[0]['name']}")
        print(f"     Args: {response1.tool_calls[0]['args']}")
    else:
        print(f"  ❌ Text response (no tool call)")
        print(f"     Content: {response1.content[:200]}...")
    print()
    print()

    # ========================================
    # TEST 2: Medium Context with explicit instructions
    # ========================================
    print("📝 TEST 2: MEDIUM CONTEXT with explicit tool calling instructions")
    print("-" * 80)
    print("Hypothesis: Explicit instructions work even with conversation history")
    print()

    test2_messages = [
        SystemMessage(content="""You are a research assistant. You have read_file tool to read files.

CRITICAL RULE: After you get a list of files, your NEXT response MUST be a read_file tool call.
DO NOT write text like "I found a file..." or "I should read..."
ONLY call read_file with the exact file path you see.
NO TEXT RESPONSES - ONLY TOOL CALLS."""),
        HumanMessage(content="Find information about coffee shops in San Francisco"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "list_all_files",
                "args": {},
                "id": "test_call_1"
            }]
        ),
        ToolMessage(
            content="Directory: /Users/cleve/dev/deep_research_from_scratch_liquidai/src/deep_research_from_scratch/files\nFiles: [FILE] coffee_shops_sf.md",
            tool_call_id="test_call_1"
        )
    ]

    print("Input messages:")
    for i, msg in enumerate(test2_messages, 1):
        msg_preview = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
        print(f"  {i}. {msg.__class__.__name__}: {msg_preview}...")
    print()

    response2 = model_with_tools.invoke(test2_messages)

    print("Response:")
    if response2.tool_calls:
        print(f"  ✅ Called tool: {response2.tool_calls[0]['name']}")
        print(f"     Args: {response2.tool_calls[0]['args']}")
    else:
        print(f"  ❌ Text response (no tool call)")
        print(f"     Content: {response2.content[:200]}...")
    print()
    print()

    # ========================================
    # TEST 3: Full Context with explicit instructions
    # ========================================
    print("📝 TEST 3: FULL CONTEXT with explicit tool calling instructions")
    print("-" * 80)
    print("Hypothesis: Explicit instructions work even with full research agent context")
    print()

    research_brief = """I want to identify and evaluate the coffee shops in San Francisco that are considered the best based specifically on coffee quality. My research should focus on analyzing and comparing coffee shops within the San Francisco area, using coffee quality as the primary criterion."""

    test3_messages = [
        SystemMessage(content="""You are a research assistant. Today's date is 2025-10-25.

**Your job:** Use tools to find and read files, then answer the user's question.

**CRITICAL - Tool Calling Rules:**
- After list_all_files returns results, your NEXT response MUST be a read_file tool call
- DO NOT write "The directory contains..." or "I found a file..."
- DO NOT explain your plan
- ONLY call read_file with the exact path from the file list
- NO TEXT until AFTER you have read the file

**Workflow:**
Step 1: list_all_files → get file paths
Step 2: read_file with exact path (NO TEXT - just tool call)
Step 3: After reading, provide your answer

**Example - CORRECT:**
[list_all_files returns: coffee.md]
Your response: [call read_file("coffee.md")] ✓

**Example - WRONG:**
[list_all_files returns: coffee.md]
Your response: "I see coffee.md, I should read it..." ✗ NEVER DO THIS

REMEMBER: Step 2 must be ONLY a tool call, no text."""),
        HumanMessage(content=research_brief),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "list_all_files",
                "args": {},
                "id": "test_call_2"
            }]
        ),
        ToolMessage(
            content="Directory: /Users/cleve/dev/deep_research_from_scratch_liquidai/src/deep_research_from_scratch/files\nFiles: [FILE] coffee_shops_sf.md",
            tool_call_id="test_call_2"
        )
    ]

    print("Input messages:")
    for i, msg in enumerate(test3_messages, 1):
        msg_preview = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
        print(f"  {i}. {msg.__class__.__name__}: {msg_preview}...")
    print()

    # Count total tokens approximately
    total_chars = sum(len(str(msg.content)) for msg in test3_messages if hasattr(msg, 'content'))
    print(f"Approximate context size: ~{total_chars} characters (~{total_chars//4} tokens)")
    print()

    response3 = model_with_tools.invoke(test3_messages)

    print("Response:")
    if response3.tool_calls:
        print(f"  ✅ Called tool: {response3.tool_calls[0]['name']}")
        print(f"     Args: {response3.tool_calls[0]['args']}")
    else:
        print(f"  ❌ Text response (no tool call)")
        print(f"     Content: {response3.content[:200]}...")
    print()
    print()

    # ========================================
    # SUMMARY
    # ========================================
    print("="*80)
    print("SUMMARY")
    print("="*80)

    test1_success = bool(response1.tool_calls)
    test2_success = bool(response2.tool_calls)
    test3_success = bool(response3.tool_calls)

    print(f"Test 1 (Minimal + Explicit):  {'✅ Tool call' if test1_success else '❌ Text response'}")
    print(f"Test 2 (Medium + Explicit):   {'✅ Tool call' if test2_success else '❌ Text response'}")
    print(f"Test 3 (Full + Explicit):     {'✅ Tool call' if test3_success else '❌ Text response'}")
    print()

    if test1_success and test2_success and test3_success:
        print("🔍 CONCLUSION: Explicit instructions work!")
        print("   → LFM2 can follow tool calling workflow with very explicit prompts")
        print("   → We should update the MCP prompts with these explicit instructions")
    elif test1_success and not test2_success:
        print("🔍 CONCLUSION: Context length still a factor despite explicit instructions")
        print("   → Even with clear rules, longer context causes failures")
        print("   → Breaking up the process into smaller sessions may help")
    elif not test1_success:
        print("🔍 CONCLUSION: Behavioral tendency cannot be overridden with instructions")
        print("   → LFM2 fundamentally prefers explanation mode over action mode")
        print("   → Need architectural changes (deterministic workflow, mega-tool, etc.)")
    elif test1_success and test2_success and not test3_success:
        print("🔍 CONCLUSION: Verbose system prompts overwhelm the model")
        print("   → Simplify system prompts dramatically")
        print("   → Keep instructions minimal and direct")
    else:
        print("🔍 CONCLUSION: Mixed results - partial success")
        print(f"   → Working: {sum([test1_success, test2_success, test3_success])}/3 tests")
        print("   → May need to experiment with prompt variations")
    print()

if __name__ == "__main__":
    asyncio.run(run_tests())
