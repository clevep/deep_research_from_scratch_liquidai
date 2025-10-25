
"""Research Agent Implementation.

This module implements a research agent that can perform iterative web searches
and synthesis to answer complex research questions.

Key features:
- Iterative web search with Tavily API
- Strategic reflection using think_tool
- Message pruning for LFM2 compatibility (keeps 2-message context for tool calling)
- Workflow state tracking to guide research process
"""

from pydantic import BaseModel, Field
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain.chat_models import init_chat_model

from deep_research_from_scratch.state_research import ResearcherState, ResearcherOutputState
from deep_research_from_scratch.utils import tavily_search, get_today_str, think_tool
from deep_research_from_scratch.prompts import research_agent_prompt, compress_research_system_prompt, compress_research_human_message

# ===== CONFIGURATION =====

# Set up tools and model binding
tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

# Initialize main research model (with tools) - uses LFM2-Tool on port 8080
model = init_chat_model(
    model="lfm2",
    model_provider="openai",
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key",
    temperature=0.2,
)
model_with_tools = model.bind_tools(tools)

# Initialize summarization model (with json_mode for structured output) - uses base LFM2 on port 8081
summarization_model = (
    init_chat_model(
        model="lfm2",
        model_provider="openai",
        base_url="http://localhost:8081/v1",
        api_key="sk-no-key",
        temperature=0.2,
    )
    .bind(
        response_format={"type": "json_object"},
        max_tokens=1024,
        extra_body={"cache_prompt": False}
    )
)

# Initialize compression model - uses base LFM2 on port 8081 for plain text generation
# LFM2 supports 32,768 tokens - using 32,000 to match original configuration
compress_model = init_chat_model(
    model="lfm2",
    model_provider="openai",
    base_url="http://localhost:8081/v1",
    api_key="sk-no-key",
    temperature=0.2,
    max_tokens=32000,
)

# ===== AGENT NODES =====

def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions with LFM2 compatibility.

    This node:
    1. Tracks research progress (search count, think_tool usage)
    2. Provides explicit workflow guidance based on state
    3. Uses message pruning to maintain 2-message context for LFM2

    Returns updated state with model response.
    """
    # MESSAGE PRUNING + WORKFLOW STATE FOR LFM2
    # Critical: LFM2 only calls tools in response to HumanMessages, not ToolMessages
    messages = state["researcher_messages"]

    # Extract research question
    research_question = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            research_question = msg.content
            break

    # Count searches performed
    search_count = sum(
        1 for m in messages 
        if hasattr(m, 'tool_calls') and m.tool_calls and
        any(tc.get('name') == 'tavily_search' for tc in m.tool_calls)
    )

    # Check if think_tool was used recently (last 2 messages)
    recent_messages = messages[-2:] if len(messages) >= 2 else messages
    has_recent_thought = any(
        hasattr(m, 'tool_calls') and m.tool_calls and
        any(tc.get('name') == 'think_tool' for tc in m.tool_calls)
        for m in recent_messages
    )

    # Build explicit workflow instructions based on state
    if search_count == 0:
        next_step = "**NEXT ACTION:** Call tavily_search with a broad query about the research topic."
    elif search_count >= 5:
        next_step = "**NEXT ACTION:** You've reached the 5 search limit. Provide your final research answer now."
    elif not has_recent_thought and search_count > 0:
        next_step = "**NEXT ACTION:** Call think_tool to reflect on your search results. DO NOT write text - ONLY call the tool."
    else:
        next_step = "**NEXT ACTION:** Based on your reflection, either: (1) Call tavily_search if you need more info, OR (2) Provide your final answer if you have enough."

    system_content = f"""{research_agent_prompt.format(date=get_today_str())}

**RESEARCH QUESTION:**
{research_question}

**PROGRESS:** {search_count}/5 searches completed

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

def tool_node(state: ResearcherState):
    """Execute all tool calls from the previous LLM response.

    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
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
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, we have a final answer
    return "compress_research"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue research loop
        "compress_research": "compress_research", # Provide final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call") # Loop back for more research
agent_builder.add_edge("compress_research", END)

# Compile the agent
researcher_agent = agent_builder.compile()
