from langgraph.types import Command
from typing_extensions import Literal
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from dotenv import load_dotenv
import os
import aiosqlite

from .utils import get_ollama_model
from .state import MyState
from .tools.handoffs import assign_to_analyst
from .tools.python_executor import execute_code
from .tools.analyst import (
    discover_models,
    csv_writer,
    make_fit_tool,
    make_model_info_tool
)
from .prompts.analyst import analyst_prompt
from .prompts.supervisor import supervisor_prompt


load_dotenv()

async def get_checkpointer():
    """
    Initialize SQLite checkpointer once at app startup.
    Returns the checkpointer instance to be reused across all graph invocations.
    """
    conn = await aiosqlite.connect("checkpoints.db")
    saver = AsyncSqliteSaver(conn)
    return saver, conn

def make_graph(
    checkpointer=None
):
    """
    Creates the graph. Reuses the same checkpointer for all invocations if provided.

    Args:
        checkpointer: Reused checkpointer instance.
    """

    # ======= SUPERVISOR =======
    supervisor_llm = get_ollama_model(
        model_name=os.getenv("SUPERVISOR_MODEL", "qwen3.8:27b"),
        temperature=0.0,
    ) 

    supervisor_agent = create_agent(
        model=supervisor_llm,
        tools = [assign_to_analyst],
        system_prompt=supervisor_prompt,
        name="agent_supervisor",
        state_schema=MyState
    )

    # ======= ANALYST AGENT =======
    llm = get_ollama_model(
        model_name=os.getenv("ANALYST_MODEL", "qwen3.8:27b"),
        temperature=0.0,
    ) 

    AVAILABLE_MODELS = discover_models()

    # Both tools are built ONCE — docstrings are static from here on
    fit_and_forecast = make_fit_tool(AVAILABLE_MODELS)
    get_model_info = make_model_info_tool(AVAILABLE_MODELS)

    tools = [execute_code, csv_writer, get_model_info, fit_and_forecast]

    analyst_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=analyst_prompt,  # System prompt for the analyst agent
        name="analyst_agent",
        state_schema=MyState,
        middleware=[
            TodoListMiddleware()
        ],
    )



    # ======= NODES =======
    # -------ANALYST AGENT NODE-------

    async def analyst_agent_node(
        state: MyState,
    ) -> Command[Literal["supervisor"]]:
        """
        Main node of the graph.
        """
        print("[GRAPH] Entering analyst_agent_node")
        # invoke the agent
        result = await analyst_agent.ainvoke({'messages' : state["messages"]})

        # get results
        last_msg = result["messages"][-1]
        code_logs = result.get("code_logs", [])
        todos = result.get("todos", [])

        # Propagate subagent's updates in the general state and route back to the supervisor for the next iteration.
        # NOTE: if you do not update todos here, the todos are not generally updated! 
        return Command(
            update={
                "messages": [HumanMessage(content=last_msg.content)],  # update messages with the last message content
                "code_logs" : code_logs,
                "todos": todos,  # propagate the todos
            },
            goto="supervisor",
        )
    
    
    # ======= GRAPH  BUILDING =======

    builder = StateGraph(MyState)
        
    builder.add_node(
        "supervisor", supervisor_agent
    )  # , destinations=("data_analyst", "simulator", END)
    builder.add_node("analyst", analyst_agent_node)

    builder.add_edge(
        START, "supervisor"
    )  # since we have Command(goto=...) everywhere, we do not need other edges.

    graph = builder.compile(checkpointer=checkpointer)

    return graph
