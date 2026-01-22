"""
LangGraph RAG pipeline implementation.

This module contains the LangGraph-based RAG pipeline with all nodes,
graph construction logic, and the RAGStreamHandler class.
"""

import re
import time
from typing import TYPE_CHECKING, Annotated, Iterator, Optional, TypedDict, Union

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langgraph.graph import END, StateGraph

from src.agents.base import AgentBase
from src.agents.comp_analyzing_agent import prompt as caa_prompt
from src.agents.intent_guard_agent import prompt as guard_prompt
from src.agents.rag_agent import prompt as rag_prompt
from src.agents.rag_agent import prompt_en as rag_prompt_en
from src.agents.universe_rag_agent import prompt as universal_rag_prompt
from src.agents.universe_rag_agent import prompt_en as universal_rag_prompt_en
from src.common.config import EmbeddingConfig, get_enable_rerank
from src.common.logger import get_logger
from src.common.time_utils import get_elapsed_tips
from src.frontend.i18n import t
from src.rag.doc_embedder import DocumentEmbedder
from src.rag.doc_processing import DocumentMeta
from src.rag.ob import (
    DEFAULT_COMPONENT,
    DEFAULT_RERANK_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    replace_doc_url,
    supported_components,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Citation processing settings
BUFFER_SIZE_THRESHOLD = 128
DOC_CITE_PATTERN = r"(\[+\@(\d+)\]+)"


# =============================================================================
# RAG State Definition
# =============================================================================

class RAGState(TypedDict):
    """
    State schema for LangGraph RAG pipeline.

    This state is passed between nodes in the graph and contains all
    information needed for the RAG processing flow.
    """

    # Input parameters
    query: str
    chat_history: list[dict]
    llm_model: str
    embedding_config: EmbeddingConfig
    suffixes: list[str]
    universal_rag: bool
    rerank: bool
    search_docs: bool
    lang: str
    show_refs: bool

    # Processing state
    start_time: float
    query_type: Optional[str]  # "Chat" or "Features"
    related_components: list[str]
    query_embedded: Optional[list[float]]
    docs: list[Document]
    rag_agent: Optional[AgentBase]  # type: ignore
    doc_embedder: Optional[DocumentEmbedder]  # type: ignore

    # Output state
    progress_messages: Annotated[list[str], lambda x, y: x + y]
    response_chunks: Annotated[list[Union[str, AIMessageChunk]], lambda x, y: x + y]
    pruned_refs: list[str]

    # Control flow
    should_early_return: bool
    mode: str  # "no_search", "universal_rag", "oceanbase_rag"


# =============================================================================
# Graph Construction
# =============================================================================

def create_rag_graph():
    """
    Create the LangGraph pipeline for RAG processing.

    The graph structure:
    1. START -> route_mode (determines which pipeline to use)
    2. route_mode -> no_search_node OR universal_rag_start OR oceanbase_rag_start
    3. oceanbase_rag_start -> analyze_intent -> route_intent
    4. route_intent -> handle_non_oceanbase OR analyze_components
    5. analyze_components -> search_documents
    6. search_documents -> generate_response
    7. universal_rag_start -> embed_query -> search_documents_universal -> generate_response
    8. generate_response -> END

    Returns:
        Compiled graph instance ready for execution
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("route_mode", _route_mode_node)
    graph.add_node("no_search_node", _no_search_node)
    graph.add_node("universal_rag_start", _universal_rag_start_node)
    graph.add_node("embed_query_universal", _embed_query_universal_node)
    graph.add_node("search_documents_universal", _search_documents_universal_node)
    graph.add_node("oceanbase_rag_start", _oceanbase_rag_start_node)
    graph.add_node("analyze_intent", _analyze_intent_node)
    graph.add_node("route_intent", _route_intent_node)
    graph.add_node("handle_non_oceanbase", _handle_non_oceanbase_node)
    graph.add_node("analyze_components", _analyze_components_node)
    graph.add_node("search_documents", _search_documents_node)
    graph.add_node("generate_response", _generate_response_node)

    # Set entry point
    graph.set_entry_point("route_mode")

    # Add edges
    graph.add_conditional_edges(
        "route_mode",
        _route_mode_condition,
        {
            "no_search": "no_search_node",
            "universal_rag": "universal_rag_start",
            "oceanbase_rag": "oceanbase_rag_start",
        },
    )

    # Universal RAG flow
    graph.add_edge("universal_rag_start", "embed_query_universal")
    graph.add_edge("embed_query_universal", "search_documents_universal")
    graph.add_edge("search_documents_universal", "generate_response")

    # OceanBase RAG flow
    graph.add_edge("oceanbase_rag_start", "analyze_intent")
    graph.add_edge("analyze_intent", "route_intent")
    graph.add_conditional_edges(
        "route_intent",
        _route_intent_condition,
        {
            "early_return": "handle_non_oceanbase",
            "continue": "analyze_components",
        },
    )
    graph.add_edge("analyze_components", "search_documents")
    graph.add_edge("search_documents", "generate_response")

    # Terminal nodes
    graph.add_edge("no_search_node", END)
    graph.add_edge("handle_non_oceanbase", END)
    graph.add_edge("generate_response", END)

    return graph.compile()


# =============================================================================
# Graph Nodes
# =============================================================================

# -----------------------------------------------------------------------------
# Routing Nodes
# -----------------------------------------------------------------------------

def _route_mode_node(state: RAGState) -> RAGState:
    """Route to appropriate pipeline based on search_docs flag."""
    if not state["search_docs"]:
        state["mode"] = "no_search"
    elif state["universal_rag"]:
        state["mode"] = "universal_rag"
    else:
        state["mode"] = "oceanbase_rag"
    return state


def _route_mode_condition(state: RAGState) -> str:
    """Conditional routing based on mode."""
    return state["mode"]


def _route_intent_node(state: RAGState) -> RAGState:
    """Route based on query intent."""
    return state


def _route_intent_condition(state: RAGState) -> str:
    """Conditional routing based on query intent."""
    if state["query_type"] == "Chat":
        return "early_return"
    return "continue"


# -----------------------------------------------------------------------------
# No-Search Mode Nodes
# -----------------------------------------------------------------------------

def _no_search_node(state: RAGState) -> RAGState:
    """Handle no-search mode - generate response without document retrieval."""
    logger.info("Document search disabled, using no-search mode")
    state["progress_messages"].append(None)  # Signal first token

    prompt = (
        universal_rag_prompt if state["lang"] == "zh" else universal_rag_prompt_en
    )
    agent = AgentBase(prompt=prompt, llm_model=state["llm_model"])
    for chunk in agent.stream(state["query"], state["chat_history"], document_snippets=""):
        state["response_chunks"].append(chunk)

    return state


# -----------------------------------------------------------------------------
# Universal RAG Mode Nodes
# -----------------------------------------------------------------------------

def _universal_rag_start_node(state: RAGState) -> RAGState:
    """Initialize universal RAG mode."""
    logger.info("Using universal RAG mode")
    if state["doc_embedder"] is None:
        state["doc_embedder"] = DocumentEmbedder(
            embedding_config=state["embedding_config"]
        )
    return state


def _embed_query_universal_node(state: RAGState) -> RAGState:
    """Embed query for universal RAG."""
    state["progress_messages"].append(
        t("embedding_query", state["lang"])
        + get_elapsed_tips(state["start_time"], lang=state["lang"])
    )
    state["query_embedded"] = state["doc_embedder"].embed_query(state["query"])
    return state


def _search_documents_universal_node(state: RAGState) -> RAGState:
    """Search documents for universal RAG."""
    state["progress_messages"].append(
        t("searching_docs", state["lang"])
        + get_elapsed_tips(state["start_time"], lang=state["lang"])
    )
    state["docs"] = state["doc_embedder"].doc_search_by_vector(
        state["query_embedded"], limit=DEFAULT_SEARCH_LIMIT
    )
    logger.info(f"Universal RAG: found {len(state['docs'])} documents")
    return state


# -----------------------------------------------------------------------------
# OceanBase RAG Mode Nodes
# -----------------------------------------------------------------------------

def _oceanbase_rag_start_node(state: RAGState) -> RAGState:
    """Initialize OceanBase RAG mode."""
    logger.info("Using OceanBase-specific RAG mode")
    if state["doc_embedder"] is None:
        state["doc_embedder"] = DocumentEmbedder(
            embedding_config=state["embedding_config"]
        )

    # Initialize RAG agent
    prompt = rag_prompt if state["lang"] == "zh" else rag_prompt_en
    state["rag_agent"] = AgentBase(prompt=prompt, llm_model=state["llm_model"])

    return state


def _analyze_intent_node(state: RAGState) -> RAGState:
    """Analyze query intent."""
    state["progress_messages"].append(
        t("analyzing_intent", state["lang"])
        + get_elapsed_tips(state["start_time"], lang=state["lang"])
    )
    logger.info(f"Analyzing query intent: query_len={len(state['query'])}")

    agent = AgentBase(prompt=guard_prompt, llm_model=state["llm_model"])
    result = agent.invoke_json(state["query"])

    state["query_type"] = (
        result.get("type", "Features") if hasattr(result, "get") else "Features"
    )
    logger.info(f"Query intent: {state['query_type']}")
    return state


def _handle_non_oceanbase_node(state: RAGState) -> RAGState:
    """Handle queries that are not OceanBase-related."""
    logger.info("Query type is Chat, not OceanBase-related")
    state["progress_messages"].append(
        t("no_oceanbase", state["lang"])
        + get_elapsed_tips(state["start_time"], lang=state["lang"])
    )

    for chunk in state["rag_agent"].stream(
        state["query"], state["chat_history"], document_snippets=""
    ):
        state["response_chunks"].append(chunk)

    return state


def _analyze_components_node(state: RAGState) -> RAGState:
    """Analyze related OceanBase components."""
    state["progress_messages"].append(
        t("analyzing_components", state["lang"])
        + get_elapsed_tips(state["start_time"], lang=state["lang"])
    )

    logger.info(
        f"Analyzing components: query_len={len(state['query'])}, "
        f"history_len={len(state['chat_history'])}"
    )

    history_text = "\n".join(
        msg["content"] for msg in state["chat_history"] if msg["role"] == "user"
    )
    combined_query = "\n".join([history_text, state["query"]])

    agent = AgentBase(prompt=caa_prompt, llm_model=state["llm_model"])
    result = agent.invoke_json(
        query=combined_query,
        background_history=[],
        supported_components=supported_components,
    )

    raw_components = (
        result.get("components", [DEFAULT_COMPONENT])
        if hasattr(result, "get")
        else [DEFAULT_COMPONENT]
    )

    # Validate components
    visited = set()
    valid = []
    for comp in raw_components:
        if comp in supported_components and comp not in visited:
            visited.add(comp)
            valid.append(comp)

    if DEFAULT_COMPONENT not in valid:
        valid.append(DEFAULT_COMPONENT)

    state["related_components"] = valid
    logger.info(f"Related components: {state['related_components']}")

    state["progress_messages"].append(
        t("list_related_components", state["lang"], ", ".join(state["related_components"]))
        + get_elapsed_tips(state["start_time"], lang=state["lang"])
    )

    return state


def _search_documents_node(state: RAGState) -> RAGState:
    """Search documents across related components."""
    rerankable = (
        (state["rerank"] or get_enable_rerank())
        and state["doc_embedder"].has_rerank()
    )
    limit = (
        DEFAULT_SEARCH_LIMIT
        if rerankable
        else max(3, 13 - 3 * len(state["related_components"]))
    )

    logger.info(
        f"Searching documents: components={state['related_components']}, "
        f"limit={limit}, rerankable={rerankable}"
    )

    state["progress_messages"].append(
        t("embedding_query", state["lang"])
        + get_elapsed_tips(state["start_time"], lang=state["lang"])
    )
    state["query_embedded"] = state["doc_embedder"].embed_query(state["query"])

    # Search each component
    all_docs = []
    for comp in state["related_components"]:
        state["progress_messages"].append(
            t("searching_docs_for", state["lang"], comp)
            + get_elapsed_tips(state["start_time"], lang=state["lang"])
        )

        comp_docs = state["doc_embedder"].doc_search_by_vector(
            state["query_embedded"],
            partition_names=[comp],
            limit=limit,
        )
        all_docs.extend(comp_docs)
        logger.debug(f"Found {len(comp_docs)} docs for {comp}")

    # Rerank if applicable
    if rerankable and len(state["related_components"]) > 1:
        state["progress_messages"].append(
            t("reranking_docs", state["lang"])
            + get_elapsed_tips(state["start_time"], lang=state["lang"])
        )
        all_docs = state["doc_embedder"].rerank(state["query"], all_docs)
        state["docs"] = all_docs[:DEFAULT_RERANK_LIMIT]
        logger.debug(f"Reranked to {len(state['docs'])} documents")
    else:
        state["docs"] = all_docs
        logger.debug(f"Using {len(state['docs'])} documents without reranking")

    return state


# -----------------------------------------------------------------------------
# Response Generation Node
# -----------------------------------------------------------------------------

def _generate_response_node(state: RAGState) -> RAGState:
    """Generate LLM response with citations and references."""
    state["progress_messages"].append(
        t("llm_thinking", state["lang"])
        + get_elapsed_tips(state["start_time"], lang=state["lang"])
    )
    logger.info(f"Generating response with {len(state['docs'])} documents")

    # Prepare document context
    docs_content = "\n=====\n".join(
        f"文档片段:\n\n{doc.page_content}" for doc in state["docs"]
    )

    # Get response iterator
    if state["universal_rag"]:
        prompt = (
            universal_rag_prompt if state["lang"] == "zh" else universal_rag_prompt_en
        )
        agent = AgentBase(prompt=prompt, llm_model=state["llm_model"])
        ans_iter = agent.stream(
            state["query"], state["chat_history"], document_snippets=docs_content
        )
    else:
        ans_iter = state["rag_agent"].stream(
            state["query"], state["chat_history"], document_snippets=docs_content
        )

    # Process stream and extract citations
    visited: dict[str, int] = {}
    pruned_refs: list[str] = []
    buffer = ""
    first_token_sent = False

    for chunk in ans_iter:
        buffer += chunk.content

        # Process citations in buffer
        if "[" in buffer and len(buffer) < BUFFER_SIZE_THRESHOLD:
            buffer, new_refs = _process_citations_in_buffer(
                buffer, visited, state["docs"]
            )
            pruned_refs.extend(new_refs)

        # Signal first token
        if not first_token_sent:
            first_token_sent = True
            state["response_chunks"].append(None)

        state["response_chunks"].append(AIMessageChunk(content=buffer))
        buffer = ""

    # Flush remaining buffer
    if buffer:
        state["response_chunks"].append(AIMessageChunk(content=buffer))

    state["pruned_refs"] = pruned_refs
    logger.debug(f"Extracted {len(pruned_refs)} references")

    # Generate references
    if state["show_refs"]:
        ref_tip = t("ref_tips", state["lang"])

        if pruned_refs:
            state["response_chunks"].append(AIMessageChunk(content="\n\n" + ref_tip))
            for ref in pruned_refs:
                state["response_chunks"].append(AIMessageChunk(content="\n" + ref))
        elif state["docs"]:
            state["response_chunks"].append(AIMessageChunk(content="\n\n" + ref_tip))
            visited_urls: dict[str, bool] = {}

            for doc in state["docs"]:
                meta = DocumentMeta.model_validate(doc.metadata)
                doc_url = replace_doc_url(meta.doc_url)

                if doc_url in visited_urls:
                    continue
                visited_urls[doc_url] = True

                count = len(visited_urls)
                ref_text = f"{count}. [{meta.doc_name}]({doc_url})"
                state["response_chunks"].append(AIMessageChunk(content="\n" + ref_text))

    # Append suffixes
    for suffix in state["suffixes"]:
        state["response_chunks"].append(AIMessageChunk(content=suffix + "\n"))

    return state


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _process_citations_in_buffer(
    buffer: str, visited: dict[str, int], docs: list[Document]
) -> tuple[str, list[str]]:
    """Process and replace citation markers in buffer."""
    new_refs = []
    matches = re.findall(DOC_CITE_PATTERN, buffer)

    if not matches:
        return buffer, new_refs

    # Sort by original text (reverse to handle overlapping replacements)
    sorted_matches = sorted(matches, key=lambda x: x[0], reverse=True)

    for original, order in sorted_matches:
        doc = docs[int(order) - 1]
        meta = DocumentMeta.model_validate(doc.metadata)
        doc_url = replace_doc_url(meta.doc_url)

        # Get or assign reference index
        if doc_url in visited:
            idx = visited[doc_url]
        else:
            idx = len(visited) + 1
            visited[doc_url] = idx
            ref_text = f"{idx}. [{meta.doc_name}]({doc_url})"
            new_refs.append(ref_text)

        # Replace citation marker
        ref_link = f"[[{idx}]]({doc_url})"
        buffer = buffer.replace(original, ref_link)

    return buffer, new_refs


# =============================================================================
# RAG Stream Handler
# =============================================================================

class RAGStreamHandler:
    """
    Handler for RAG (Retrieval-Augmented Generation) streaming responses.

    LangGraph Pipeline Structure:
    ============================

    The workflow is organized as a directed graph with conditional routing:

    ```
                            START
                              |
                              v
                        [route_mode]
                              |
        ┌─────────────────────┴─────────────────────────────┐
        |                     |                             |
        v                     v                             v
    (no_search)        (universal_rag)               (oceanbase_rag)
        |                     |                             |
        |                     |                             v
        |                     |                    [oceanbase_rag_start]
        |                     v                             |
        |           [universal_rag_start]                   v
        |                     |                      [analyze_intent]
        |                     |                             |
        |                     v                             v
        |            [embed_query_universal]            [route_intent]
        |                     |                             |
        |                     |                    ┌────────┴────────┐
        |                     v                    |                 |
        |           [search_documents_universal]   v                 v
        |                     |               (continue)     (early_return) 
        |                     |                    |                 |
        |                     |                    |                 v
        |                     |          [analyze_components] [handle_non_oceanbase] 
        |                     |                    |                 |
        |                     |                    v                 |
        |                     |            [search_documents]        |
        |                     |                    |                 |
        |                     |                    |                 |
        |                     v                    v                 |
        |                   [    generate_response   ]               |
        |                               |                            |
        └───────────────────────────────┴────────────────────────────┘
                                        |
                                        v
                                      END
    ```

    Pipeline Modes:
    --------------

    1. **No-Search Mode** (search_docs=False):
       - route_mode → no_search_node → generate_response → END
       - Directly generates response without document retrieval

    2. **Universal RAG Mode** (universal_rag=True):
       - route_mode → universal_rag_start → embed_query_universal →
         search_documents_universal → generate_response → END
       - Simple embedding-based search across all documents

    3. **OceanBase RAG Mode** (default, universal_rag=False):
       - route_mode → oceanbase_rag_start → analyze_intent → route_intent
       - If query_type == "Chat": route_intent → handle_non_oceanbase → END
       - If query_type == "Features": route_intent → analyze_components →
         search_documents → generate_response → END
       - Component-aware search with optional reranking

    State Management:
    ----------------
    The pipeline uses a shared RAGState that flows through all nodes, containing:
    - Input parameters (query, chat_history, llm_model, etc.)
    - Processing state (query_type, related_components, docs, etc.)
    - Output state (progress_messages, response_chunks, pruned_refs)

    Example:
        handler = RAGStreamHandler(
            query="How to optimize OceanBase?",
            chat_history=[],
            llm_model="gpt-4",
            embedding_config=EmbeddingConfig.from_env(),
        )
        for chunk in handler.stream():
            if isinstance(chunk, str):
                print(f"Progress: {chunk}")
            elif chunk is not None:
                print(chunk.content, end="")
    """

    def __init__(
        self,
        query: str,
        chat_history: list[dict],
        llm_model: str,
        embedding_config: EmbeddingConfig,
        *,
        suffixes: Optional[list[str]] = None,
        universal_rag: bool = False,
        rerank: bool = False,
        search_docs: bool = True,
        lang: str = "zh",
        show_refs: bool = True,
    ):
        """
        Initialize the RAG stream handler.

        Args:
            query: User query string.
            chat_history: List of previous chat messages.
            llm_model: Name of the LLM model to use.
            embedding_config: Embedding model configuration.
            suffixes: Additional suffixes to append to response.
            universal_rag: Whether to use universal RAG mode.
            rerank: Whether to enable document reranking.
            search_docs: Whether to search documents.
            lang: Language code ("zh" or "en").
            show_refs: Whether to show reference list.
        """
        self.query = query
        self.chat_history = chat_history
        self.llm_model = llm_model
        self.suffixes = suffixes or []
        self.universal_rag = universal_rag
        self.rerank = rerank
        self.search_docs = search_docs
        self.lang = lang
        self.show_refs = show_refs

        # Internal state
        self._start_time: float = 0
        self._docs: list[Document] = []
        self._rag_agent: Optional[AgentBase] = None

        # Initialize document embedder
        self._doc_embedder = DocumentEmbedder(embedding_config=embedding_config)

    def stream(self) -> Iterator[Union[str, AIMessageChunk]]:
        """
        Execute the RAG pipeline using LangGraph and stream responses.

        This method initializes the LangGraph state and executes the compiled graph.
        The graph processes the query through the appropriate pipeline based on
        configuration (no_search, universal_rag, or oceanbase_rag mode).

        Execution Flow:
        1. Initialize RAGState with input parameters
        2. Create and compile the LangGraph pipeline
        3. Execute the graph using graph.invoke()
        4. Yield progress messages and response chunks from final state

        The graph execution follows the workflow defined in create_rag_graph(),
        with state flowing through nodes and conditional edges determining the path.

        Yields:
            Progress messages (str) and response chunks (AIMessageChunk).
            Progress messages indicate current processing stage (e.g., "Analyzing intent...")
            Response chunks contain the streaming LLM response with citations.

        Note:
            Currently uses synchronous graph.invoke(). Future versions may support
            async streaming via graph.astream() for real-time progress updates.
        """
        self._start_time = time.time()
        logger.info(
            f"Starting RAG stream: query_len={len(self.query)}, "
            f"universal_rag={self.universal_rag}, rerank={self.rerank}, "
            f"search_docs={self.search_docs}, lang={self.lang}"
        )

        # Initialize state for LangGraph
        initial_state: RAGState = {
            "query": self.query,
            "chat_history": self.chat_history,
            "llm_model": self.llm_model,
            "embedding_config": self._doc_embedder._embedding_config,
            "suffixes": self.suffixes,
            "universal_rag": self.universal_rag,
            "rerank": self.rerank,
            "search_docs": self.search_docs,
            "lang": self.lang,
            "show_refs": self.show_refs,
            "start_time": self._start_time,
            "query_type": None,
            "related_components": [],
            "query_embedded": None,
            "docs": [],
            "rag_agent": None,
            "doc_embedder": self._doc_embedder,
            "progress_messages": [],
            "response_chunks": [],
            "pruned_refs": [],
            "should_early_return": False,
            "mode": "",
        }

        # Create and run the graph
        graph = create_rag_graph()
        final_state = graph.invoke(initial_state)

        # Yield progress messages and response chunks
        for msg in final_state["progress_messages"]:
            if msg is not None:
                yield msg

        for chunk in final_state["response_chunks"]:
            yield chunk

        # Log completion
        elapsed = time.time() - self._start_time
        logger.info(f"RAG stream completed in {elapsed:.2f} seconds")
