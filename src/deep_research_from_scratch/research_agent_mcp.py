
"""Research Agent with MCP Integration.

This module implements a research agent that integrates with Model Context Protocol (MCP)
servers to access tools and resources. The agent demonstrates how to use MCP filesystem
server for local document research and analysis.

Key features:
- MCP server integration for tool access
- Async operations for concurrent tool execution (required by MCP protocol)
- Filesystem operations for local document research
- Secure directory access with permission checking
- Research compression for efficient processing
- Lazy MCP client initialization for LangGraph Platform compatibility
- Message pruning for LFM2 compatibility (keeps 2-message context for tool calling)
"""

import os
import re

from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain_core.tools import tool as langchain_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END

from deep_research_from_scratch.prompts import research_agent_prompt_with_mcp, compress_research_system_prompt, compress_research_human_message
from deep_research_from_scratch.state_research import ResearcherState, ResearcherOutputState
from deep_research_from_scratch.utils import get_today_str, think_tool, get_current_dir

# ===== CONFIGURATION =====

# MCP server configuration for filesystem access
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",  # Auto-install if needed
            "@modelcontextprotocol/server-filesystem",
            str(get_current_dir() / "files")  # Path to research documents
        ],
        "transport": "stdio"  # Communication via stdin/stdout
    }
}

# Global client variable - will be initialized lazily
_client = None

def get_mcp_client():
    """Get or initialize MCP client lazily to avoid issues with LangGraph Platform."""
    global _client
    if _client is None:
        _client = MultiServerMCPClient(mcp_config)
    return _client

# Initialize models - using dual LFM2 setup
# Compression model uses base LFM2 on port 8081 for plain text generation
compress_model = init_chat_model(
    model="lfm2",
    model_provider="openai",
    base_url="http://localhost:8081/v1",
    api_key="sk-no-key",
    temperature=0.2,
    max_tokens=32000,
)

# Main research model uses LFM2-Tool on port 8080 for tool calling
model = init_chat_model(
    model="lfm2",
    model_provider="openai",
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key",
    temperature=0,
)

# ===== COMPOSITE TOOLS =====

async def list_all_available_files_impl() -> str:
    """Composite tool that lists all available files in one operation.

    Combines list_allowed_directories + list_directory into a single tool call,
    reducing the number of sequential decisions the LLM needs to make.
    """
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    tools_by_name = {tool.name: tool for tool in mcp_tools}

    # Get allowed directories
    list_dirs_tool = tools_by_name.get("list_allowed_directories")
    if not list_dirs_tool:
        return "Error: list_allowed_directories tool not found"

    dirs_result = await list_dirs_tool.ainvoke({})

    # Parse the directories (result is a string like "Allowed directories:\n/path/to/files")
    paths = re.findall(r'/[^\n]+', str(dirs_result))

    if not paths:
        return "No directories found"

    # List files in each directory
    list_dir_tool = tools_by_name.get("list_directory")
    if not list_dir_tool:
        return f"Allowed directories: {paths[0]}\n(list_directory tool not available)"

    all_files = []
    for path in paths:
        try:
            files_result = await list_dir_tool.ainvoke({"path": path})
            # Parse file names and build full paths so LFM2 can copy exact paths
            # files_result format: "[FILE] filename.ext\n[FILE] other.ext"
            file_entries = []
            for line in str(files_result).split('\n'):
                if '[FILE]' in line:
                    filename = line.replace('[FILE]', '').strip()
                    if filename:
                        full_path = f"{path}/{filename}"
                        file_entries.append(f"  - {filename} (FULL_PATH: {full_path})")

            if file_entries:
                all_files.append(f"\nDirectory: {path}\nFiles:\n" + "\n".join(file_entries))
            else:
                all_files.append(f"\nDirectory: {path}\nFiles: {files_result}")
        except Exception as e:
            all_files.append(f"\nDirectory: {path}\nError: {str(e)}")

    return "".join(all_files)

@langchain_tool
async def list_all_files() -> str:
    """List all available files in the research directory.

    This is a composite tool that automatically:
    1. Finds allowed directories
    2. Lists all files in those directories

    Use this as your FIRST tool call to see what files are available.

    Returns: String listing all available files with their full paths
    """
    return await list_all_available_files_impl()

# ===== AGENT NODES =====

async def llm_call(state: ResearcherState):
    """Analyze current state and decide on tool usage with MCP integration.

    This node:
    1. Retrieves available tools from MCP server
    2. Binds tools to the language model
    3. Processes user input and decides on tool usage
    4. Uses message pruning + workflow state tracking for LFM2

    Returns updated state with model response.
    """
    # Get available tools from MCP server
    client = get_mcp_client()
    mcp_tools = await client.get_tools()

    # Create simplified tool set to reduce cognitive load on LFM2
    read_file_tool = next((t for t in mcp_tools if t.name in ["read_file", "read_text_file"]), None)

    if read_file_tool:
        tools = [list_all_files, read_file_tool, think_tool]
    else:
        # Fallback to all MCP tools if read_file not found
        tools = [list_all_files] + mcp_tools + [think_tool]

    # Initialize model with tool binding
    model_with_tools = model.bind_tools(tools)

    # MESSAGE PRUNING + WORKFLOW STATE FOR LFM2
    # Critical: LFM2 only calls tools in response to HumanMessages, not ToolMessages
    messages = state["researcher_messages"]

    # Extract research question
    research_question = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            research_question = msg.content
            break

    # Determine workflow state based on tool calls made
    has_listed = any(
        hasattr(m, 'tool_calls') and m.tool_calls and 
        any(tc.get('name') == 'list_all_files' for tc in m.tool_calls) 
        for m in messages
    )
    has_read = any(
        hasattr(m, 'tool_calls') and m.tool_calls and 
        any(tc.get('name') in ['read_file', 'read_text_file'] for tc in m.tool_calls)
        for m in messages
    )

    # Build explicit workflow instructions based on state
    if not has_listed:
        next_step = "**NEXT ACTION:** Call list_all_files tool."
    elif not has_read:
        next_step = "**NEXT ACTION:** Call read_file tool. Copy the FULL_PATH exactly from the file list. DO NOT write text - ONLY call the tool."
    else:
        next_step = "**NEXT ACTION:** You have read the file. Provide your final research answer."

    system_content = f"""{research_agent_prompt_with_mcp.format(date=get_today_str())}

**RESEARCH QUESTION:**
{research_question}

{next_step}

CRITICAL: Make tool calls. Do NOT write explanations."""

    # KEY FIX: Convert ToolMessage to HumanMessage format
    # LFM2 only reliably calls tools in response to HumanMessages (proven in tests)
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, ToolMessage):
            # Convert ToolMessage content to HumanMessage so LFM2 treats it as a request
            pruned_messages = [HumanMessage(content=last_msg.content)]
        elif isinstance(last_msg, HumanMessage):
            pruned_messages = [last_msg]
        else:
            # For other message types, wrap content as HumanMessage
            pruned_messages = [HumanMessage(content=str(getattr(last_msg, 'content', '')))]
    else:
        pruned_messages = []

    return {
        "researcher_messages": [
            model_with_tools.invoke([SystemMessage(content=system_content)] + pruned_messages)
        ]
    }

async def tool_node(state: ResearcherState):
    """Execute tool calls using MCP tools.

    This node:
    1. Retrieves current tool calls from the last message
    2. Executes all tool calls using async operations (required for MCP)
    3. Returns formatted tool results

    Note: MCP requires async operations due to inter-process communication
    with the MCP server subprocess. This is unavoidable.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    async def execute_tools():
        """Execute all tool calls. MCP tools require async execution."""
        # Get fresh tool references from MCP server
        client = get_mcp_client()
        mcp_tools = await client.get_tools()
        tools = [list_all_files] + mcp_tools + [think_tool]
        tools_by_name = {tool.name: tool for tool in tools}

        # Execute tool calls (sequentially for reliability)
        observations = []
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            if tool_call["name"] == "think_tool":
                # think_tool is sync, use regular invoke
                observation = tool.invoke(tool_call["args"])
            else:
                # MCP tools are async, use ainvoke
                observation = await tool.ainvoke(tool_call["args"])
            observations.append(observation)

        # Format results as tool messages
        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for observation, tool_call in zip(observations, tool_calls)
        ]

        return tool_outputs

    messages = await execute_tools()

    return {"researcher_messages": messages}

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for further processing or reporting.

    This function filters out think_tool calls and focuses on substantive
    file-based research content from MCP tools.
    """

    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]

    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue with tool execution or compress research.

    Determines whether to continue with tool execution or compress research
    based on whether the LLM made tool calls.
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # Continue to tool execution if tools were called
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, compress research findings
    return "compress_research"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder_mcp = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder_mcp.add_node("llm_call", llm_call)
agent_builder_mcp.add_node("tool_node", tool_node)
agent_builder_mcp.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder_mcp.add_edge(START, "llm_call")
agent_builder_mcp.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",        # Continue to tool execution
        "compress_research": "compress_research",  # Compress research findings
    },
)
agent_builder_mcp.add_edge("tool_node", "llm_call")  # Loop back for more processing
agent_builder_mcp.add_edge("compress_research", END)

# Compile the agent
agent_mcp = agent_builder_mcp.compile()
