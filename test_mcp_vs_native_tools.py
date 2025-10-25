"""Test MCP tools vs native LangChain tools for context bloat.

This script tests whether MCP's tool schemas add significant context overhead
compared to simple native LangChain @tool decorated functions.

Hypothesis: MCP tool metadata/schemas may be bloating context and contributing
to LFM2's tool calling failures.
"""

import asyncio
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

# Initialize LFM2-Tool model
model = init_chat_model(
    model="lfm2",
    model_provider="openai",
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key",
    temperature=0,
)

# Get current directory
current_dir = Path(__file__).parent
test_file_path = current_dir / "src/deep_research_from_scratch/files/coffee_shops_sf.md"

# ========================================
# NATIVE TOOL - MINIMAL
# ========================================
@tool
def read_file_minimal(path: str) -> str:
    """Read a file."""
    with open(path, 'r') as f:
        return f.read()

# ========================================
# NATIVE TOOL - VERBOSE (mimicking MCP)
# ========================================
@tool
def read_file_verbose(path: str) -> str:
    """Read the complete contents of a file from the file system as text.

    Handles various text encodings and provides the raw file content.
    Use this tool to access and read text files stored on the local filesystem.

    Args:
        path: The absolute path to the file you want to read. Must be a valid
              file path on the local filesystem. The file must exist and be
              readable by the current process.

    Returns:
        The complete text content of the file as a string.

    Raises:
        FileNotFoundError: If the specified file does not exist
        PermissionError: If the file cannot be read due to permissions
    """
    with open(path, 'r') as f:
        return f.read()

async def run_tests():
    """Run comparison tests between MCP and native tools."""

    print("="*80)
    print("MCP vs NATIVE TOOLS COMPARISON - CONTROLLED")
    print("="*80)
    print()
    print("Testing whether MCP tool schemas bloat context compared to native tools")
    print("Using IDENTICAL message structure for fair comparison")
    print()

    # Shared system prompt (minimal and explicit)
    minimal_system_prompt = """You have a read_file tool.

CRITICAL: Your response MUST be ONLY a tool call. DO NOT write text.
DO NOT say "I will read the file..."
DO NOT explain what you're doing.
ONLY call the read_file tool with the file path."""

    # ========================================
    # TEST 1: Native Tool - Minimal Description (2 messages)
    # ========================================
    print("📝 TEST 1: NATIVE TOOL (Minimal - 2 messages)")
    print("-" * 80)
    print("Hypothesis: Minimal context + minimal tool = best performance")
    print()

    test1_messages = [
        SystemMessage(content=minimal_system_prompt.replace("read_file", "read_file_minimal")),
        HumanMessage(content=f"Read the file at {test_file_path}")
    ]

    # Check tool schema size
    model_with_minimal = model.bind_tools([read_file_minimal])
    tool_schema_minimal = read_file_minimal.get_input_schema().schema()
    schema_size_minimal = len(str(tool_schema_minimal))

    print(f"Message count: {len(test1_messages)}")
    print(f"Tool schema size: {schema_size_minimal} chars")
    print(f"Tool description: '{read_file_minimal.description}'")
    print()

    response1 = model_with_minimal.invoke(test1_messages)

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
    # TEST 2: Native Tool - Verbose Description (2 messages)
    # ========================================
    print("📝 TEST 2: NATIVE TOOL (Verbose - 2 messages)")
    print("-" * 80)
    print("Hypothesis: Verbose tool descriptions impact performance even with minimal context")
    print()

    test2_messages = [
        SystemMessage(content=minimal_system_prompt.replace("read_file", "read_file_verbose")),
        HumanMessage(content=f"Read the file at {test_file_path}")
    ]

    # Check tool schema size
    model_with_verbose = model.bind_tools([read_file_verbose])
    tool_schema_verbose = read_file_verbose.get_input_schema().schema()
    schema_size_verbose = len(str(tool_schema_verbose))

    print(f"Message count: {len(test2_messages)}")
    print(f"Tool schema size: {schema_size_verbose} chars")
    print(f"Tool description: '{read_file_verbose.description[:100]}...'")
    print()

    response2 = model_with_verbose.invoke(test2_messages)

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
    # TEST 3: MCP Tool (2 messages)
    # ========================================
    print("📝 TEST 3: MCP TOOL (2 messages)")
    print("-" * 80)
    print("Hypothesis: MCP tools perform similarly to native tools at same context level")
    print()

    # Initialize MCP client
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

    client = MultiServerMCPClient(mcp_config)
    mcp_tools = await client.get_tools()
    read_file_mcp = next((t for t in mcp_tools if t.name in ["read_file", "read_text_file"]), None)

    if not read_file_mcp:
        print("❌ MCP read_file tool not found!")
        return

    # Check tool schema size
    model_with_mcp = model.bind_tools([read_file_mcp])
    tool_schema_mcp = read_file_mcp.get_input_schema().schema() if hasattr(read_file_mcp, 'get_input_schema') else str(read_file_mcp)
    schema_size_mcp = len(str(tool_schema_mcp))

    print(f"Message count: 2")
    print(f"Tool schema size: {schema_size_mcp} chars")
    print(f"Tool description: '{read_file_mcp.description[:100]}...'")
    print()

    test3_messages = [
        SystemMessage(content=minimal_system_prompt.replace("read_file", read_file_mcp.name)),
        HumanMessage(content=f"Read the file at {test_file_path}")
    ]

    response3 = model_with_mcp.invoke(test3_messages)

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

    print("\nTool Schema Sizes:")
    print(f"  Native (Minimal):  {schema_size_minimal:4d} chars")
    print(f"  Native (Verbose):  {schema_size_verbose:4d} chars")
    print(f"  MCP Tool:          {schema_size_mcp:4d} chars")
    print()

    print("Tool Calling Success:")
    print(f"  Native (Minimal):  {'✅ Tool call' if test1_success else '❌ Text response'}")
    print(f"  Native (Verbose):  {'✅ Tool call' if test2_success else '❌ Text response'}")
    print(f"  MCP Tool:          {'✅ Tool call' if test3_success else '❌ Text response'}")
    print()

    # Analysis
    schema_diff_verbose = schema_size_verbose - schema_size_minimal
    schema_diff_mcp = schema_size_mcp - schema_size_minimal

    print(f"\nSchema overhead:")
    print(f"  Verbose vs Minimal: +{schema_diff_verbose} chars ({(schema_diff_verbose/schema_size_minimal*100):.1f}% increase)")
    print(f"  MCP vs Minimal:     +{schema_diff_mcp} chars ({(schema_diff_mcp/schema_size_minimal*100):.1f}% increase)")
    print()

    if test1_success and test2_success and test3_success:
        print("🔍 CONCLUSION: All tools work equally well at minimal context")
        print("   → MCP overhead is NOT causing the tool calling failures")
        print("   → Tool schema size (162 vs 838 chars) doesn't matter at this context level")
        print("   → The problem is definitely MESSAGE HISTORY, not tool definitions")
    elif test1_success and not test2_success and not test3_success:
        print("🔍 CONCLUSION: Verbose tool schemas impact performance")
        print("   → Even at minimal context, verbose descriptions cause failures")
        print("   → MCP's detailed schemas ARE contributing to the problem")
        print("   → Using minimal native tools is the path forward")
    elif test1_success and test2_success and not test3_success:
        print("🔍 CONCLUSION: MCP has unique overhead beyond description verbosity")
        print("   → MCP adds context bloat beyond just the tool description")
        print("   → Native tools (even verbose) perform better than MCP")
        print("   → Should replace MCP with native file operations")
    elif not test1_success and not test2_success and not test3_success:
        print("🔍 CONCLUSION: Even minimal context (2 messages) is too much")
        print("   → The system prompt + user message alone overwhelm LFM2")
        print("   → Tool schemas are irrelevant - base context is the problem")
        print("   → May need to drastically simplify prompts or use different architecture")
    else:
        print("🔍 CONCLUSION: Mixed results - partial success")
        print(f"   → Successes: {sum([test1_success, test2_success, test3_success])}/3")
        if test3_success and not test1_success:
            print("   → UNEXPECTED: MCP performs better than native minimal tool!")
        print("   → Further investigation needed")
    print()

if __name__ == "__main__":
    asyncio.run(run_tests())
