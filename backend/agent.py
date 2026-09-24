"""
LangGraph Agent Workflow & Tool Routing

Coordinates user intent analysis, tool selection
(Calculator, Document RAG, Web Search),
and direct LLM generation with multi-turn memory.

Primary chat generation is handled through OpenRouter.
"""

import re
import logging
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from backend.tools.calculator import calculate, preprocess_expression
from backend.tools.document_tool import query_document
from backend.tools.web_search import search_web
from backend.services.document_service import get_active_document_info
from backend.memory import get_session_history_as_messages
from backend.llm import invoke_llm_safely
from backend.config import is_openrouter_configured


logger = logging.getLogger(__name__)


# ============================================================
# AGENT STATE
# ============================================================

class AgentState(TypedDict):
    conversation_id: str
    user_query: str
    intent: str
    tool_input: Optional[str]
    tool_output: Optional[Any]
    tool_used: str
    activity_steps: List[str]
    final_response: str


# ============================================================
# INTENT DETECTION PATTERNS
# ============================================================

# Regular expression heuristics for safe math detection
MATH_PATTERN = re.compile(
    r'(?:'
    r'^\s*[\(\d\.\+\-\*\/\%\^\s]+\s*$|'
    r'\bcalculate\b|\bcompute\b|\bsolve\b|'
    r'\d+(?:\.\d+)?\s*%\s+of\s+\d+|'
    r'\bwhat\s+is\s+[\d\.\s\+\-\*\/\%\^\(\)]+\??$|'
    r'\bsqrt\s*\(\s*\d+\s*\)'
    r')',
    re.IGNORECASE
)


# Words indicating document queries
DOCUMENT_KEYWORDS = [
    "pdf",
    "document",
    "uploaded",
    "paper",
    "file",
    "textbook",
    "summarize this",
    "summary of this",
    "main points of the pdf",
    "according to the document",
    "in this document",
    "from the pdf",
]


# Words indicating live search queries
SEARCH_KEYWORDS = [
    "search the web",
    "search online",
    "latest news",
    "current price",
    "what happened today",
    "current information about",
    "who won yesterday",
]


# ============================================================
# INTENT ANALYSIS NODE
# ============================================================

def analyze_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyze the user's request and determine
    whether an external tool is required.
    """

    query = state["user_query"].strip()

    activity = list(
        state.get("activity_steps", [])
    )

    activity.append("Request received")

    lower_query = query.lower()


    # --------------------------------------------------------
    # 1. DOCUMENT / PDF INTENT
    # --------------------------------------------------------

    doc_info = get_active_document_info()

    is_doc_query = any(
        keyword in lower_query
        for keyword in DOCUMENT_KEYWORDS
    )

    # If a document is active, detect common
    # document-related follow-up requests.
    if doc_info.get("has_document") and (
        "summarize" in lower_query
        or "key takeaways" in lower_query
        or "main points" in lower_query
    ):
        is_doc_query = True


    if is_doc_query:

        activity.append(
            "Intent identified: Document query"
        )

        activity.append(
            "Document retrieval tool selected"
        )

        return {
            "intent": "document",
            "tool_input": query,
            "tool_used": "document",
            "activity_steps": activity,
        }


    # --------------------------------------------------------
    # 2. MATHEMATICAL / CALCULATOR INTENT
    # --------------------------------------------------------

    is_math = False

    clean_calc = preprocess_expression(query)

    # If preprocessing produces only numbers and
    # mathematical operators, treat it as calculation.
    if (
        re.match(
            r'^[\d\.\s\+\-\*\/\(\)\%\^]+$',
            clean_calc
        )
        and any(
            operator in clean_calc
            for operator in "+-*/%^"
        )
    ):
        is_math = True

    elif MATH_PATTERN.search(query):

        # Additional safety check:
        # query should contain numbers and operators.
        if (
            any(char.isdigit() for char in query)
            and any(
                operator in query
                for operator in "+-*/%^x×÷"
            )
        ):
            is_math = True


    if is_math:

        activity.append(
            "Intent identified: Mathematical calculation"
        )

        activity.append(
            "Calculator tool selected"
        )

        return {
            "intent": "calculator",
            "tool_input": query,
            "tool_used": "calculator",
            "activity_steps": activity,
        }


    # --------------------------------------------------------
    # 3. WEB SEARCH INTENT
    # --------------------------------------------------------

    if any(
        keyword in lower_query
        for keyword in SEARCH_KEYWORDS
    ):

        activity.append(
            "Intent identified: Real-time search query"
        )

        activity.append(
            "Web search tool selected"
        )

        return {
            "intent": "web_search",
            "tool_input": query,
            "tool_used": "web_search",
            "activity_steps": activity,
        }


    # --------------------------------------------------------
    # 4. DEFAULT: GENERAL PURPOSE OPENROUTER LLM
    # --------------------------------------------------------

    activity.append(
        "Intent identified: General inquiry"
    )

    activity.append(
        "Direct AI response selected"
    )

    return {
        "intent": "direct_llm",
        "tool_input": None,
        "tool_used": "none",
        "activity_steps": activity,
    }


# ============================================================
# CONDITIONAL ROUTER
# ============================================================

def route_intent(state: AgentState) -> str:
    """
    Route execution based on the detected intent.
    """

    return state["intent"]


# ============================================================
# CALCULATOR NODE
# ============================================================

def calculator_node(
    state: AgentState
) -> Dict[str, Any]:
    """
    Execute the safe mathematical calculator.
    """

    activity = list(
        state.get("activity_steps", [])
    )

    calc_res = calculate(
        state["tool_input"]
        or state["user_query"]
    )

    activity.append(
        "Calculation evaluated safely"
    )


    if calc_res["success"]:

        ans = (
            f"The answer is "
            f"**{calc_res['result']}**."
        )

        if (
            calc_res["expression"]
            != state["user_query"].strip()
        ):
            ans = (
                f"Calculation: "
                f"`{calc_res['expression']}`\n\n"
                f"Result: "
                f"**{calc_res['result']}**"
            )

    else:

        ans = calc_res["result"]


    activity.append("Response generated")

    return {
        "tool_output": calc_res,
        "final_response": ans,
        "activity_steps": activity,
    }


# ============================================================
# DOCUMENT RAG NODE
# ============================================================

def document_node(
    state: AgentState
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks from an uploaded
    document and synthesize an answer using OpenRouter.
    """

    activity = list(
        state.get("activity_steps", [])
    )

    doc_res = query_document(
        state["tool_input"]
        or state["user_query"]
    )


    # --------------------------------------------------------
    # NO DOCUMENT UPLOADED
    # --------------------------------------------------------

    if not doc_res["has_document"]:

        activity.append(
            "No active document found"
        )

        activity.append(
            "Response generated"
        )

        return {
            "tool_output": doc_res,
            "final_response": (
                "No document has been uploaded yet. "
                "Please use the **Upload PDF** button "
                "in the sidebar to upload a document first."
            ),
            "activity_steps": activity,
        }


    activity.append(
        f"Retrieved {doc_res['chunks_found']} "
        f"relevant context chunks from "
        f"'{doc_res['filename']}'"
    )


    # --------------------------------------------------------
    # RAG + OPENROUTER
    # --------------------------------------------------------

    if (
        is_openrouter_configured()
        and doc_res["chunks_found"] > 0
    ):

        activity.append(
            "Synthesizing document answer with OpenRouter"
        )


        prompt = (
            "You are answering a question based strictly "
            "on the following uploaded document: "
            f"'{doc_res['filename']}'.\n\n"

            "=== DOCUMENT CONTEXT ===\n"
            f"{doc_res['context']}\n"
            "========================\n\n"

            f"User Question: "
            f"{state['user_query']}\n\n"

            "Instructions:\n"

            "1. Answer based on the document context "
            "provided above.\n"

            "2. If the context does not contain enough "
            "information, politely say: "
            "'I couldn't find enough information about "
            "that in the uploaded document.'\n"

            "3. Mention relevant page numbers if helpful.\n"

            "4. Keep the explanation accurate, clear, "
            "and professional."
        )


        rag_messages = [

            SystemMessage(
                content=(
                    "You are an intelligent "
                    "document analysis assistant."
                )
            ),

            HumanMessage(
                content=prompt
            ),
        ]


        answer, success = invoke_llm_safely(
            rag_messages
        )


        if success:

            activity.append(
                "Response generated"
            )

        else:

            activity.append(
                "AI service error"
            )


    # --------------------------------------------------------
    # FALLBACK WITHOUT OPENROUTER
    # --------------------------------------------------------

    else:

        if doc_res["chunks_found"] == 0:

            answer = (
                "I couldn't find enough information "
                "about that in the uploaded document."
            )

        else:

            answer = (
                f"Relevant information from "
                f"**{doc_res['filename']}**:\n\n"
                f"{doc_res['context']}"
            )

        activity.append(
            "Response generated"
        )


    return {
        "tool_output": doc_res,
        "final_response": answer,
        "activity_steps": activity,
    }


# ============================================================
# WEB SEARCH NODE
# ============================================================

def web_search_node(
    state: AgentState
) -> Dict[str, Any]:
    """
    Execute the web-search tool and optionally
    synthesize the search results with OpenRouter.
    """

    activity = list(
        state.get("activity_steps", [])
    )


    search_res = search_web(
        state["tool_input"]
        or state["user_query"]
    )


    activity.append(
        "Web search lookup processed"
    )


    # --------------------------------------------------------
    # WEB SEARCH NOT CONFIGURED
    # --------------------------------------------------------

    if not search_res.get("configured"):

        final_answer = (
            "Live web search is not configured "
            "for this project. "
            "To enable live internet search, configure "
            "`TAVILY_API_KEY` in the `.env` file."
        )


    # --------------------------------------------------------
    # SEARCH SUCCESSFUL
    # --------------------------------------------------------

    elif search_res.get("success"):

        if is_openrouter_configured():

            activity.append(
                "Synthesizing search results with OpenRouter"
            )

            prompt = (
                "Synthesize an answer based on these "
                "web search results for the query: "
                f"'{state['user_query']}'.\n\n"

                "Search Results:\n"
                f"{search_res['result']}"
            )


            final_answer, success = (
                invoke_llm_safely(
                    [
                        HumanMessage(
                            content=prompt
                        )
                    ]
                )
            )


            if success:

                activity.append(
                    "Response generated"
                )

            else:

                activity.append(
                    "AI service error"
                )


        else:

            final_answer = (
                "Web Search Results:\n"
                f"{search_res['result']}"
            )

            activity.append(
                "Response generated"
            )


    # --------------------------------------------------------
    # SEARCH FAILED
    # --------------------------------------------------------

    else:

        final_answer = search_res.get(
            "result",
            "Search request could not be completed."
        )

        activity.append(
            "Response generated"
        )


    return {
        "tool_output": search_res,
        "final_response": final_answer,
        "activity_steps": activity,
    }


# ============================================================
# DIRECT GENERAL-PURPOSE AI NODE
# ============================================================

def direct_llm_node(
    state: AgentState
) -> Dict[str, Any]:
    """
    Generate a general-purpose AI response using
    OpenRouter with multi-turn conversation memory.
    """

    activity = list(
        state.get("activity_steps", [])
    )

    activity.append(
        "Processing conversation memory"
    )


    # Retrieve previous conversation messages
    # so the assistant can understand context.
    history_messages = (
        get_session_history_as_messages(
            state["conversation_id"]
        )
    )


    # Append the current user message.
    history_messages.append(
        HumanMessage(
            content=state["user_query"]
        )
    )


    activity.append(
        "Synthesizing answer with OpenRouter"
    )


    # --------------------------------------------------------
    # OPENROUTER CONFIGURATION CHECK
    # --------------------------------------------------------

    if not is_openrouter_configured():

        final_answer = (
            "AI service is temporarily unavailable. "
            "Please check your OpenRouter API "
            "configuration.\n\n"

            "Make sure `OPENROUTER_API_KEY` is "
            "configured in the `.env` file, "
            "then restart the backend server."
        )

        activity.append(
            "AI service error"
        )


    # --------------------------------------------------------
    # GENERATE OPENROUTER RESPONSE
    # --------------------------------------------------------

    else:

        final_answer, success = (
            invoke_llm_safely(
                history_messages
            )
        )


        if success:

            activity.append(
                "Response generated"
            )

        else:

            activity.append(
                "AI service error"
            )


    return {
        "final_response": final_answer,
        "activity_steps": activity,
    }


# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_agent_graph():
    """
    Construct and compile the LangGraph StateGraph workflow.
    """

    builder = StateGraph(
        AgentState
    )


    # --------------------------------------------------------
    # ADD NODES
    # --------------------------------------------------------

    builder.add_node(
        "analyze_intent",
        analyze_intent_node
    )

    builder.add_node(
        "calculator",
        calculator_node
    )

    builder.add_node(
        "document",
        document_node
    )

    builder.add_node(
        "web_search",
        web_search_node
    )

    builder.add_node(
        "direct_llm",
        direct_llm_node
    )


    # --------------------------------------------------------
    # START -> INTENT ANALYSIS
    # --------------------------------------------------------

    builder.add_edge(
        START,
        "analyze_intent"
    )


    # --------------------------------------------------------
    # CONDITIONAL ROUTING
    # --------------------------------------------------------

    builder.add_conditional_edges(

        "analyze_intent",

        route_intent,

        {
            "calculator": "calculator",
            "document": "document",
            "web_search": "web_search",
            "direct_llm": "direct_llm",
        }
    )


    # --------------------------------------------------------
    # TERMINAL EDGES
    # --------------------------------------------------------

    builder.add_edge(
        "calculator",
        END
    )

    builder.add_edge(
        "document",
        END
    )

    builder.add_edge(
        "web_search",
        END
    )

    builder.add_edge(
        "direct_llm",
        END
    )


    return builder.compile()


# ============================================================
# COMPILED AGENT
# ============================================================

AGENT_GRAPH = build_agent_graph()


# ============================================================
# PUBLIC AGENT ENTRYPOINT
# ============================================================

def run_agent(
    conversation_id: str,
    user_query: str
) -> Dict[str, Any]:
    """
    Public entrypoint for running the
    LangChain / LangGraph agent.
    """

    initial_state: AgentState = {

        "conversation_id":
            conversation_id,

        "user_query":
            user_query,

        "intent":
            "direct_llm",

        "tool_input":
            None,

        "tool_output":
            None,

        "tool_used":
            "none",

        "activity_steps":
            [],

        "final_response":
            "",
    }


    try:

        final_state = (
            AGENT_GRAPH.invoke(
                initial_state
            )
        )


        return {

            "answer":
                final_state.get(
                    "final_response",
                    ""
                ),

            "tool_used":
                final_state.get(
                    "tool_used",
                    "none"
                ),

            "activity":
                final_state.get(
                    "activity_steps",
                    []
                ),
        }


    except Exception as exc:

        logger.error(
            "Agent execution error: %s",
            exc,
            exc_info=True
        )


        return {

            "answer": (
                "Something went wrong while "
                "generating the answer. "
                "Please try again."
            ),

            "tool_used":
                "none",

            "activity": [
                "Request received",
                "Execution encountered an error",
            ],
        }