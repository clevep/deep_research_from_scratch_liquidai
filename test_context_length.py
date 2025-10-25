"""Test script to determine if LFM2's issue is context length or behavioral tendency.

This script tests the model's behavior with different context lengths to isolate
whether the problem is:
A) Context length overwhelming the model
B) Behavioral tendency to explain rather than act

Updated to use VERBOSE tool descriptions based on finding that verbose descriptions
actually help LFM2 understand when to call tools.
"""

import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
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
test_file_path = current_dir / "src/deep_research_from_scratch/files/coffee_shops_sf.md"

# ========================================
# VERBOSE TOOL DEFINITION
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

@tool
def list_all_files_verbose() -> str:
    """List all available files in the research directory.

    This tool provides a comprehensive listing of files available for research.
    Use this tool to discover what files are available before reading them.

    Returns:
        A formatted string listing all available files with their full paths.
        Each file is listed on a separate line with its complete path so you
        can use the exact path with read_file_verbose.
    """
    files_dir = current_dir / "src/deep_research_from_scratch/files"
    files = list(files_dir.glob("*.md"))
    if not files:
        return "No files found in research directory"

    result = f"Directory: {files_dir}\nFiles:\n"
    for f in files:
        result += f"  - {f.name} (full path: {f})\n"
    return result

async def run_tests():
    """Run controlled experiments with different context lengths using VERBOSE tools."""

    # Bind verbose tools to model
    model_with_tools = model.bind_tools([read_file_verbose, list_all_files_verbose])

    print("="*80)
    print("CONTEXT LENGTH EXPERIMENT WITH VERBOSE TOOLS - 5 RUNS PER TEST")
    print("="*80)
    print()
    print("Testing if verbose tool descriptions help maintain tool calling with longer context")
    print("Running each test 5 times to check for consistency (temperature=0)")
    print()

    # ========================================
    # TEST 1: Minimal Context with verbose tool (5 runs)
    # ========================================
    print("📝 TEST 1: MINIMAL CONTEXT (2 messages) with verbose tool")
    print("-" * 80)
    print("Hypothesis: Verbose tool descriptions enable successful tool calling")
    print()

    test1_messages = [
        SystemMessage(content="""You have a read_file_verbose tool.

CRITICAL: Your response MUST be ONLY a tool call. DO NOT write text.
DO NOT say "I will read the file..."
DO NOT explain what you're doing.
ONLY call the read_file_verbose tool with the file path."""),
        HumanMessage(content=f"Read the file at {test_file_path}")
    ]

    print("Input messages:")
    for i, msg in enumerate(test1_messages, 1):
        print(f"  {i}. {msg.__class__.__name__}: {msg.content[:100]}...")
    print()

    # Run 5 times
    test1_results = []
    print("Running 5 times:")
    for run in range(1, 6):
        response = model_with_tools.invoke(test1_messages)
        success = bool(response.tool_calls)
        test1_results.append(success)
        result_str = "✅ Tool call" if success else "❌ Text response"
        print(f"  Run {run}: {result_str}")

    test1_success_count = sum(test1_results)
    print(f"\nResult: {test1_success_count}/5 successful ({test1_success_count/5*100:.0f}%)")
    print()
    print()

    # ========================================
    # TEST 2: Medium Context (4 messages) with verbose tool (5 runs)
    # ========================================
    print("📝 TEST 2: MEDIUM CONTEXT (4 messages) with verbose tool")
    print("-" * 80)
    print("Hypothesis: Verbose tool helps even with conversation history")
    print()

    test2_messages = [
        SystemMessage(content="""You are a research assistant. You have read_file_verbose tool to read files.

CRITICAL RULE: After you get a list of files, your NEXT response MUST be a read_file_verbose tool call.
DO NOT write text like "I found a file..." or "I should read..."
ONLY call read_file_verbose with the exact file path you see.
NO TEXT RESPONSES - ONLY TOOL CALLS."""),
        HumanMessage(content="Find information about coffee shops in San Francisco"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "list_all_files_verbose",
                "args": {},
                "id": "test_call_1"
            }]
        ),
        ToolMessage(
            content=f"Directory: {current_dir}/src/deep_research_from_scratch/files\nFiles:\n  - coffee_shops_sf.md (full path: {test_file_path})",
            tool_call_id="test_call_1"
        )
    ]

    print("Input messages:")
    for i, msg in enumerate(test2_messages, 1):
        msg_preview = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
        print(f"  {i}. {msg.__class__.__name__}: {msg_preview}...")
    print()

    # Run 5 times
    test2_results = []
    print("Running 5 times:")
    for run in range(1, 6):
        response = model_with_tools.invoke(test2_messages)
        success = bool(response.tool_calls)
        test2_results.append(success)
        result_str = "✅ Tool call" if success else "❌ Text response"
        print(f"  Run {run}: {result_str}")

    test2_success_count = sum(test2_results)
    print(f"\nResult: {test2_success_count}/5 successful ({test2_success_count/5*100:.0f}%)")
    print()
    print()

    # ========================================
    # TEST 3: Full Context (4 messages, verbose content) with verbose tool (5 runs)
    # ========================================
    print("📝 TEST 3: FULL CONTEXT (4 messages, verbose) with verbose tool")
    print("-" * 80)
    print("Hypothesis: Verbose tool helps even with detailed research brief context")
    print()

    research_brief = """I want to identify and evaluate the coffee shops in San Francisco that are considered the best based specifically on coffee quality. My research should focus on analyzing and comparing coffee shops within the San Francisco area, using coffee quality as the primary criterion."""

    test3_messages = [
        SystemMessage(content="""You are a research assistant. Today's date is 2025-10-25.

**Your job:** Use tools to find and read files, then answer the user's question.

**CRITICAL - Tool Calling Rules:**
- After list_all_files_verbose returns results, your NEXT response MUST be a read_file_verbose tool call
- DO NOT write "The directory contains..." or "I found a file..."
- DO NOT explain your plan
- ONLY call read_file_verbose with the exact path from the file list
- NO TEXT until AFTER you have read the file

**Workflow:**
Step 1: list_all_files_verbose → get file paths
Step 2: read_file_verbose with exact path (NO TEXT - just tool call)
Step 3: After reading, provide your answer

**Example - CORRECT:**
[list_all_files_verbose returns: coffee.md]
Your response: [call read_file_verbose("coffee.md")] ✓

**Example - WRONG:**
[list_all_files_verbose returns: coffee.md]
Your response: "I see coffee.md, I should read it..." ✗ NEVER DO THIS

REMEMBER: Step 2 must be ONLY a tool call, no text."""),
        HumanMessage(content=research_brief),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "list_all_files_verbose",
                "args": {},
                "id": "test_call_2"
            }]
        ),
        ToolMessage(
            content=f"Directory: {current_dir}/src/deep_research_from_scratch/files\nFiles:\n  - coffee_shops_sf.md (full path: {test_file_path})",
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

    # Run 5 times
    test3_results = []
    print("Running 5 times:")
    for run in range(1, 6):
        response = model_with_tools.invoke(test3_messages)
        success = bool(response.tool_calls)
        test3_results.append(success)
        result_str = "✅ Tool call" if success else "❌ Text response"
        print(f"  Run {run}: {result_str}")

    test3_success_count = sum(test3_results)
    print(f"\nResult: {test3_success_count}/5 successful ({test3_success_count/5*100:.0f}%)")
    print()
    print()

    # ========================================
    # SUMMARY
    # ========================================
    print("="*80)
    print("SUMMARY (5 runs per test)")
    print("="*80)
    print()

    print(f"Test 1 (2 msg + Verbose Tool):   {test1_success_count}/5 successful ({test1_success_count/5*100:.0f}%)")
    print(f"Test 2 (4 msg + Verbose Tool):   {test2_success_count}/5 successful ({test2_success_count/5*100:.0f}%)")
    print(f"Test 3 (4 msg + Verbose + Full): {test3_success_count}/5 successful ({test3_success_count/5*100:.0f}%)")
    print()

    # Analyze consistency
    test1_consistent = test1_success_count == 0 or test1_success_count == 5
    test2_consistent = test2_success_count == 0 or test2_success_count == 5
    test3_consistent = test3_success_count == 0 or test3_success_count == 5

    print("Consistency (temperature=0):")
    print(f"  Test 1: {'✅ Deterministic' if test1_consistent else '❌ Random/Inconsistent'}")
    print(f"  Test 2: {'✅ Deterministic' if test2_consistent else '❌ Random/Inconsistent'}")
    print(f"  Test 3: {'✅ Deterministic' if test3_consistent else '❌ Random/Inconsistent'}")
    print()

    # Use majority rule for conclusions
    test1_mostly_succeeds = test1_success_count >= 3
    test2_mostly_succeeds = test2_success_count >= 3
    test3_mostly_succeeds = test3_success_count >= 3

    if test1_mostly_succeeds and test2_mostly_succeeds and test3_mostly_succeeds:
        print("🔍 CONCLUSION: Verbose tool descriptions solve the problem!")
        print("   → LFM2 can maintain tool calling even with conversation history")
        print("   → The detailed tool descriptions act as helpful prompting")
        print("   → We can keep MCP (which has verbose descriptions) and it will work")
        if not (test1_consistent and test2_consistent and test3_consistent):
            print("   ⚠️  WARNING: Some randomness detected - not fully deterministic")
    elif test1_mostly_succeeds and test2_mostly_succeeds and not test3_mostly_succeeds:
        print("🔍 CONCLUSION: Verbose tools help but full context still overwhelms")
        print("   → Can handle 4 messages with moderate content")
        print("   → Detailed research briefs push it over the edge")
        print("   → May need to simplify user messages or filter message history")
    elif test1_mostly_succeeds and not test2_mostly_succeeds:
        print("🔍 CONCLUSION: Conversation history still a limiting factor")
        print("   → Verbose tools help with 2 messages but fail with 4")
        print("   → Need to prune message history between tool calls")
        print("   → Breaking up the process into smaller sessions required")
    elif not test1_mostly_succeeds:
        print("🔍 CONCLUSION: Even verbose tools can't help at minimal context")
        print("   → There's a deeper issue beyond tool descriptions")
        print("   → May be a prompt structure or model configuration problem")
    else:
        print("🔍 CONCLUSION: Highly inconsistent/random results")
        print("   → With temperature=0, results should be deterministic")
        print("   → Inconsistency suggests model is on a decision boundary")
        print("   → Small variations in context push it different directions")
    print()

if __name__ == "__main__":
    asyncio.run(run_tests())
