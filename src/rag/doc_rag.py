import re
import time
from typing import Iterator, Optional, Union

from langchain_core.messages import AIMessageChunk
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from sqlalchemy import Column, Integer

from src.agents.base import AgentBase
from src.agents.comp_analyzing_agent import prompt as caa_prompt
from src.agents.intent_guard_agent import prompt as guard_prompt
from src.agents.rag_agent import prompt as rag_prompt
from src.agents.rag_agent import prompt_en as rag_prompt_en
from src.agents.universe_rag_agent import prompt as universal_rag_prompt
from src.agents.universe_rag_agent import prompt_en as universal_rag_prompt_en
from src.common.config import (
    get_echo,
    get_enable_rerank,
    get_ollama_token,
    get_ollama_url,
    get_openai_embedding_api_key,
    get_openai_embedding_base_url,
    get_openai_embedding_model,
    get_table_name,
)
from src.common.connection import connection_args
from src.common.logger import get_logger
from src.frontend.i18n import t
from src.rag.documents import Document, DocumentMeta
from src.rag.documents import component_mapping as cm
from src.rag.embedding import get_embedding

logger = get_logger(__name__)

# Configuration constants
DEFAULT_TABLE_NAME = "corpus"
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_RERANK_LIMIT = 10
DEFAULT_COMPONENT = "observer"
DOC_CITE_PATTERN = r"(\[+\@(\d+)\]+)"
BUFFER_SIZE_THRESHOLD = 128

embeddings = get_embedding(
    ollama_url=get_ollama_url(),
    ollama_token=get_ollama_token(),
    base_url=get_openai_embedding_base_url(),
    api_key=get_openai_embedding_api_key(),
    model=get_openai_embedding_model(),
)

logger.info("Initializing OceanbaseVectorStore")
vs = OceanbaseVectorStore(
    embedding_function=embeddings,
    table_name=get_table_name(),
    connection_args=connection_args,
    metadata_field="metadata",
    extra_columns=[Column("component_code", Integer, primary_key=True)],
    echo=get_echo(),
)
logger.info(f"OceanbaseVectorStore initialized with table: {get_table_name()}")

doc_cite_pattern = r"(\[+\@(\d+)\]+)"


def doc_search(
    query: str,
    partition_names: Optional[list[str]] = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[Document]:
    """
    Search for documents related to the query using text similarity.

    Args:
        query: Search query text
        partition_names: Optional list of partition names to search in
        limit: Maximum number of documents to return

    Returns:
        List of relevant documents
    """
    logger.info(f"Searching documents: query length={len(query)}, partition_names={partition_names}, limit={limit}")
    docs = vs.similarity_search(
        query=query,
        k=limit,
        partition_names=partition_names,
    )
    logger.debug(f"Found {len(docs)} documents")
    return docs


def doc_search_by_vector(
    vector: list[float],
    partition_names: Optional[list[str]] = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[Document]:
    """
    Search for documents related to the query using vector similarity.

    Args:
        vector: Query embedding vector
        partition_names: Optional list of partition names to search in
        limit: Maximum number of documents to return

    Returns:
        List of relevant documents
    """
    logger.debug(f"Searching documents by vector: vector_dim={len(vector)}, partition_names={partition_names}, limit={limit}")
    docs = vs.similarity_search_by_vector(
        embedding=vector,
        k=limit,
        partition_names=partition_names,
    )
    logger.debug(f"Found {len(docs)} documents")
    return docs


# Supported components
supported_components = cm.keys()

# URL replacement patterns for documentation links
URL_REPLACERS = [
    (r"^.*oceanbase-doc", "https://gitee.com/oceanbase-devhub/oceanbase-doc/blob/V4.3.4"),
    (r"^.*ocp-doc", "https://github.com/oceanbase/ocp-doc/blob/V4.3.0"),
    (r"^.*odc-doc", "https://github.com/oceanbase/odc-doc/blob/V4.3.1"),
    (r"^.*oms-doc", "https://github.com/oceanbase/oms-doc/blob/V4.2.5"),
    (r"^.*obd-doc", "https://github.com/oceanbase/obd-doc/blob/V2.10.1"),
    (
        r"^.*oceanbase-proxy-doc",
        "https://github.com/oceanbase/oceanbase-proxy-doc/blob/V4.3.0",
    ),
    (r"^.*?ob-operator/", "https://github.com/oceanbase/ob-operator/blob/master/"),
]


def replace_doc_url(doc_url: str) -> str:
    """
    Replace documentation URL with the appropriate base URL.

    Args:
        doc_url: Original document URL

    Returns:
        Replaced document URL
    """
    for pattern, base_url in URL_REPLACERS:
        doc_url = re.sub(pattern, base_url, doc_url)
    return doc_url


def get_elapsed_tips(
    start_time: float,
    end_time: Optional[float] = None,
    /,
    lang: str = "zh",
) -> str:
    """
    Get elapsed time message.

    Args:
        start_time: Start timestamp
        end_time: End timestamp (defaults to current time)
        lang: Language code

    Returns:
        Formatted elapsed time message
    """
    end_time = end_time or time.time()
    elapsed_time = end_time - start_time
    return t("time_elapse", lang, elapsed_time)


def extract_users_input(history: list[dict]) -> str:
    """
    Extract all user messages from chat history.

    Args:
        history: List of message dictionaries

    Returns:
        Concatenated user input text
    """
    return "\n".join([msg["content"] for msg in history if msg["role"] == "user"])


def _handle_no_search_mode(
    query: str,
    chat_history: list[dict],
    llm_model: str,
    lang: str,
) -> Iterator[AIMessageChunk]:
    """
    Handle RAG streaming when document search is disabled.

    Args:
        query: User query
        chat_history: Chat history
        llm_model: LLM model name
        lang: Language code

    Yields:
        Response chunks
    """
    yield None
    universal_prompt = universal_rag_prompt if lang == "zh" else universal_rag_prompt_en
    universal_rag_agent = AgentBase(prompt=universal_prompt, llm_model=llm_model)
    ans_itr = universal_rag_agent.stream(query, chat_history, document_snippets="")
    for chunk in ans_itr:
        yield chunk


def _analyze_query_intent(query: str, llm_model: str) -> str:
    """
    Analyze the intent of the query to determine if it's OceanBase-related.

    Args:
        query: User query
        llm_model: LLM model name

    Returns:
        Query type ("Chat" or "Features")
    """
    logger.info(f"Analyzing query intent: query length={len(query)}, model={llm_model}")
    iga = AgentBase(prompt=guard_prompt, llm_model=llm_model)
    guard_res = iga.invoke_json(query)
    intent_type = guard_res.get("type", "Features") if hasattr(guard_res, "get") else "Features"
    logger.info(f"Query intent analysis result: {intent_type}")
    return intent_type


def _analyze_related_components(
    query: str,
    chat_history: list[dict],
    llm_model: str,
) -> list[str]:
    """
    Analyze which OceanBase components are related to the query.

    Args:
        query: User query
        chat_history: Chat history
        llm_model: LLM model name

    Returns:
        List of related component names
    """
    logger.info(f"Analyzing related components: query length={len(query)}, history length={len(chat_history)}, model={llm_model}")
    history_text = extract_users_input(chat_history)
    caa_agent = AgentBase(prompt=caa_prompt, llm_model=llm_model)
    analyze_res = caa_agent.invoke_json(
        query="\n".join([history_text, query]),
        background_history=[],
        supported_components=supported_components,
    )
    if hasattr(analyze_res, "get"):
        related_comps: list[str] = analyze_res.get("components", [DEFAULT_COMPONENT])
    else:
        related_comps = [DEFAULT_COMPONENT]

    # Clean up and validate components
    visited = set()
    valid_components = []
    for comp in related_comps:
        if comp in supported_components and comp not in visited:
            visited.add(comp)
            valid_components.append(comp)

    # Ensure observer is always included
    if DEFAULT_COMPONENT not in valid_components:
        valid_components.append(DEFAULT_COMPONENT)

    logger.info(f"Related components analysis result: {valid_components}")
    return valid_components


def _search_documents_by_components(
    query: str,
    query_embedded: list[float],
    related_comps: list[str],
    rerankable: bool,
    start_time: float,
    lang: str,
) -> Iterator[Union[str, list[Document]]]:
    """
    Search for documents across related components.

    Args:
        query: Original query text (for reranking)
        query_embedded: Query embedding vector
        related_comps: List of related component names
        rerankable: Whether reranking is available
        start_time: Start timestamp for progress updates
        lang: Language code

    Yields:
        Progress messages (str) and finally returns list of documents
    """
    limit = DEFAULT_SEARCH_LIMIT if rerankable else max(3, 13 - 3 * len(related_comps))
    total_docs = []
    logger.info(f"Searching documents by components: related_comps={related_comps}, limit={limit}, rerankable={rerankable}")

    yield t("embedding_query", lang) + get_elapsed_tips(start_time, lang=lang)

    for comp in related_comps:
        yield t("searching_docs_for", lang, comp) + get_elapsed_tips(start_time, lang=lang)
        comp_docs = doc_search_by_vector(
            query_embedded,
            partition_names=[comp],
            limit=limit,
        )
        total_docs.extend(comp_docs)
        logger.debug(f"Found {len(comp_docs)} documents for component {comp}")

    if rerankable and len(related_comps) > 1:
        logger.info(f"Reranking {len(total_docs)} documents")
        yield t("reranking_docs", lang) + get_elapsed_tips(
            start_time,
            lang=lang,
        )
        total_docs = embeddings.rerank(query, total_docs)
        logger.debug(f"Reranking completed, returning top {DEFAULT_RERANK_LIMIT} documents")
        yield total_docs[:DEFAULT_RERANK_LIMIT]
    else:
        logger.debug(f"Returning {len(total_docs)} documents without reranking")
        yield total_docs


def _process_response_stream(
    ans_itr: Iterator[AIMessageChunk],
    docs: list[Document],
) -> Iterator[Union[AIMessageChunk, tuple]]:
    """
    Process the response stream and extract document citations.

    Args:
        ans_itr: Iterator of response chunks
        docs: List of retrieved documents

    Yields:
        Processed response chunks with citations, and finally a tuple with references
    """
    visited = {}
    count = 0
    buffer: str = ""
    pruned_references = []
    get_first_token = False
    whole = ""

    for chunk in ans_itr:
        whole += chunk.content
        buffer += chunk.content
        if "[" in buffer and len(buffer) < BUFFER_SIZE_THRESHOLD:
            matches = re.findall(DOC_CITE_PATTERN, buffer)
            if len(matches) > 0:
                sorted(matches, key=lambda x: x[0], reverse=True)
                for m, order in matches:
                    doc = docs[int(order) - 1]
                    meta = DocumentMeta.model_validate(doc.metadata)
                    doc_name = meta.doc_name
                    doc_url = replace_doc_url(meta.doc_url)
                    idx = count + 1
                    if doc_url in visited:
                        idx = visited[doc_url]
                    else:
                        visited[doc_url] = idx
                        doc_text = f"{idx}. [{doc_name}]({doc_url})"
                        pruned_references.append(doc_text)
                        count += 1

                    ref_text = f"[[{idx}]]({doc_url})"
                    buffer = buffer.replace(m, ref_text)

        if not get_first_token:
            get_first_token = True
            yield None
        yield AIMessageChunk(content=buffer)
        buffer = ""

    if len(buffer) > 0:
        yield AIMessageChunk(content=buffer)

    # Yield references as a special marker
    yield ("_references", pruned_references)


def _generate_references(
    docs: list[Document],
    pruned_references: list[str],
    lang: str,
) -> Iterator[AIMessageChunk]:
    """
    Generate reference list from documents.

    Args:
        docs: List of documents
        pruned_references: Pre-processed references
        lang: Language code

    Yields:
        Reference chunks
    """
    ref_tip = t("ref_tips", lang)

    if len(pruned_references) > 0:
        yield AIMessageChunk(content="\n\n" + ref_tip)
        for ref in pruned_references:
            yield AIMessageChunk(content="\n" + ref)
    elif len(docs) > 0:
        yield AIMessageChunk(content="\n\n" + ref_tip)
        visited = {}
        for doc in docs:
            meta = DocumentMeta.model_validate(doc.metadata)
            doc_name = meta.doc_name
            doc_url = replace_doc_url(meta.doc_url)
            if doc_url in visited:
                continue
            visited[doc_url] = True
            count = len(visited)
            doc_text = f"{count}. [{doc_name}]({doc_url})"
            yield AIMessageChunk(content="\n" + doc_text)


def doc_rag_stream(
    query: str,
    chat_history: list[dict],
    llm_model: str,
    suffixes: list[any] = [],
    universal_rag: bool = False,
    rerank: bool = False,
    search_docs: bool = True,
    lang: str = "zh",
    show_refs: bool = True,
    **kwargs,
) -> Iterator[Union[str, AIMessageChunk]]:
    """
    Stream the response from the RAG model.

    This function orchestrates the RAG pipeline:
    1. Query intent analysis (if not universal RAG)
    2. Component analysis (if OceanBase-specific)
    3. Document search and retrieval
    4. Reranking (optional)
    5. LLM response generation
    6. Reference extraction and formatting

    Args:
        query: User query string
        chat_history: List of previous messages
        llm_model: LLM model name
        suffixes: Additional suffixes to append
        universal_rag: Whether to use universal RAG mode
        rerank: Whether to rerank documents
        search_docs: Whether to search documents
        lang: Language code ("zh" or "en")
        show_refs: Whether to show references
        **kwargs: Additional parameters

    Yields:
        Progress messages (str) and response chunks (AIMessageChunk)
    """
    start_time = time.time()
    logger.info(f"Starting doc_rag_stream: query length={len(query)}, universal_rag={universal_rag}, rerank={rerank}, search_docs={search_docs}, lang={lang}")

    # Handle case when document search is disabled
    if not search_docs:
        logger.info("Document search disabled, using no-search mode")
        for chunk in _handle_no_search_mode(query, chat_history, llm_model, lang):
            yield chunk
        return

    # OceanBase-specific RAG mode
    if not universal_rag:
        logger.info("Using OceanBase-specific RAG mode")
        # Analyze query intent
        yield t("analyzing_intent", lang) + get_elapsed_tips(start_time, start_time, lang=lang)
        query_type = _analyze_query_intent(query, llm_model)
        prompt = rag_prompt if lang == "zh" else rag_prompt_en
        rag_agent = AgentBase(prompt=prompt, llm_model=llm_model)

        # Handle non-OceanBase queries
        if query_type == "Chat":
            logger.info("Query type is Chat, not OceanBase-related")
            yield t("no_oceanbase", lang) + get_elapsed_tips(start_time, lang=lang)
            yield None
            for chunk in rag_agent.stream(query, chat_history, document_snippets=""):
                yield chunk
            return

        # Analyze related components
        yield t("analyzing_components", lang) + get_elapsed_tips(start_time, lang=lang)
        related_comps = _analyze_related_components(query, chat_history, llm_model)
        yield t(
            "list_related_components",
            lang,
            ", ".join(related_comps),
        ) + get_elapsed_tips(start_time, lang=lang)

        # Check if reranking is available
        rerankable = (rerank or get_enable_rerank()) and getattr(
            embeddings, "rerank", None
        ) is not None

        # Search documents by components
        query_embedded = embeddings.embed_query(query)
        search_results = _search_documents_by_components(
            query, query_embedded, related_comps, rerankable, start_time, lang
        )
        # Extract progress messages and final docs list
        docs = None
        for result in search_results:
            if isinstance(result, list):
                docs = result
            else:
                yield result
        if docs is None:
            docs = []

    # Universal RAG mode
    else:
        logger.info("Using universal RAG mode")
        yield t("embedding_query", lang) + get_elapsed_tips(start_time, lang=lang)
        query_embedded = embeddings.embed_query(query)

        yield t("searching_docs", lang) + get_elapsed_tips(start_time, lang=lang)
        docs = doc_search_by_vector(
            query_embedded,
            limit=DEFAULT_SEARCH_LIMIT,
        )
        logger.info(f"Universal RAG mode: found {len(docs)} documents")

    # Generate LLM response
    yield t("llm_thinking", lang) + get_elapsed_tips(start_time, lang=lang)
    logger.info(f"Generating LLM response with {len(docs)} documents")

    docs_content = "\n=====\n".join([f"文档片段:\n\n" + chunk.page_content for chunk in docs])

    if universal_rag:
        universal_prompt = universal_rag_prompt if lang == "zh" else universal_rag_prompt_en
        universal_rag_agent = AgentBase(prompt=universal_prompt, llm_model=llm_model)
        ans_itr = universal_rag_agent.stream(query, chat_history, document_snippets=docs_content)
    else:
        ans_itr = rag_agent.stream(query, chat_history, document_snippets=docs_content)

    # Process response stream and extract references
    logger.debug("Processing response stream and extracting references")
    pruned_references = []
    for chunk in _process_response_stream(ans_itr, docs):
        if isinstance(chunk, tuple) and chunk[0] == "_references":
            pruned_references = chunk[1]
            logger.debug(f"Extracted {len(pruned_references)} references")
            continue
        yield chunk

    # Generate and yield references if enabled
    if show_refs:
        logger.debug(f"Generating references: show_refs={show_refs}, pruned_references={len(pruned_references)}, docs={len(docs)}")
        for ref_chunk in _generate_references(docs, pruned_references, lang):
            yield ref_chunk

    # Append suffixes
    for suffix in suffixes:
        yield AIMessageChunk(content=suffix + "\n")
    
    elapsed = time.time() - start_time
    logger.info(f"doc_rag_stream completed in {elapsed:.2f} seconds")
