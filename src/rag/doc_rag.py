import re
import os
import time

from src.frontend.i18n import t
from typing import Iterator, Optional, Union
from langchain_core.messages import AIMessageChunk
from src.rag.embeddings import get_embedding
from src.rag.documents import Document, DocumentMeta, component_mapping as cm
from src.agents.base import AgentBase
from src.agents.rag_agent import prompt as rag_prompt, prompt_en as rag_prompt_en
from src.agents.universe_rag_agent import (
    prompt as universal_rag_prompt,
    prompt_en as universal_rag_prompt_en,
)
from src.agents.intent_guard_agent import prompt as guard_prompt
from src.agents.comp_analyzing_agent import prompt as caa_prompt
from src.common.connection import connection_args
from sqlalchemy import Column, Integer

from langchain_oceanbase.vectorstores import OceanbaseVectorStore

embeddings = get_embedding(
    ollama_url=os.getenv("OLLAMA_URL") or None,
    ollama_token=os.getenv("OLLAMA_TOKEN") or None,
    base_url=os.getenv("OPENAI_EMBEDDING_BASE_URL") or None,
    api_key=os.getenv("OPENAI_EMBEDDING_API_KEY") or os.getenv("API_KEY") or None,
    model=os.getenv("OPENAI_EMBEDDING_MODEL") or None,
)


vs = OceanbaseVectorStore(
    embedding_function=embeddings,
    table_name=os.getenv("TABLE_NAME", "corpus"),
    connection_args=connection_args,
    metadata_field="metadata",
    extra_columns=[Column("component_code", Integer, primary_key=True)],
    echo=os.getenv("ECHO") == "true",
)

doc_cite_pattern = r"(\[+\@(\d+)\]+)"


def doc_search(
    query: str,
    partition_names: Optional[list[str]] = None,
    limit: int = 10,
) -> list[Document]:
    """
    Search for documents related to the query.
    """
    docs = vs.similarity_search(
        query=query,
        k=limit,
        partition_names=partition_names,
    )

    return docs


def doc_search_by_vector(
    vector: list[float],
    partition_names: Optional[list[str]] = None,
    limit: int = 10,
) -> list[Document]:
    """
    Search for documents related to the query.
    """
    docs = vs.similarity_search_by_vector(
        embedding=vector,
        k=limit,
        partition_names=partition_names,
    )

    return docs


supported_components = cm.keys()

replacers = [
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
    Replace the doc url.
    """
    for pattern, base_url in replacers:
        doc_url = re.sub(pattern, base_url, doc_url)
    return doc_url


def get_elapsed_tips(
    start_time: float,
    end_time: Optional[float] = None,
    /,
    lang: str = "zh",
) -> str:
    end_time = end_time or time.time()
    elapsed_time = end_time - start_time
    return t("time_elapse", lang, elapsed_time)


def extract_users_input(history: list[dict]) -> str:
    """
    Extract the user's input from the chat history.
    """
    return "\n".join([msg["content"] for msg in history if msg["role"] == "user"])


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
    """
    start_time = time.time()
    if not search_docs:
        yield None
        universal_prompt = (
            universal_rag_prompt if lang == "zh" else universal_rag_prompt_en
        )
        universal_rag_agent = AgentBase(prompt=universal_prompt, llm_model=llm_model)
        ans_itr = universal_rag_agent.stream(query, chat_history, document_snippets="")
        for chunk in ans_itr:
            yield chunk
        return

    if not universal_rag:
        yield t("analyzing_intent", lang) + get_elapsed_tips(
            start_time, start_time, lang=lang
        )
        iga = AgentBase(prompt=guard_prompt, llm_model=llm_model)
        guard_res = iga.invoke_json(query)
        if hasattr(guard_res, "get"):
            query_type = guard_res.get("type", "Features")
        else:
            query_type = "Features"
        prompt = rag_prompt if lang == "zh" else rag_prompt_en
        rag_agent = AgentBase(prompt=prompt, llm_model=llm_model)
        if query_type == "Chat":
            yield t("no_oceanbase", lang) + get_elapsed_tips(start_time, lang=lang)
            yield None
            for chunk in rag_agent.stream(query, chat_history, document_snippets=""):
                yield chunk
            return
        else:
            yield t("analyzing_components", lang) + get_elapsed_tips(
                start_time, lang=lang
            )
            history_text = extract_users_input(chat_history)
            caa_agent = AgentBase(prompt=caa_prompt, llm_model=llm_model)
            analyze_res = caa_agent.invoke_json(
                query="\n".join([history_text, query]),
                background_history=[],
                supported_components=supported_components,
            )
            if hasattr(analyze_res, "get"):
                related_comps: list[str] = analyze_res.get("components", ["observer"])
            else:
                related_comps = ["observer"]

            visited = set()
            for comp in related_comps:
                if comp not in supported_components:
                    related_comps.remove(comp)
                    continue
                if comp in visited:
                    related_comps.remove(comp)
                    continue
                visited.add(comp)

            if "observer" not in related_comps:
                related_comps.append("observer")

            yield t(
                "list_related_components",
                lang,
                ", ".join(related_comps),
            ) + get_elapsed_tips(start_time, lang=lang)

            docs = []

            rerankable = (
                rerank or os.getenv("ENABLE_RERANK", None) == "true"
            ) and getattr(embeddings, "rerank", None) is not None

            limit = 10 if rerankable else max(3, 13 - 3 * len(related_comps))
            total_docs = []

            yield t("embedding_query", lang) + get_elapsed_tips(start_time, lang=lang)
            query_embedded = embeddings.embed_query(query)

            for comp in related_comps:
                yield t("searching_docs_for", lang, comp) + get_elapsed_tips(
                    start_time, lang=lang
                )
                total_docs.extend(
                    doc_search_by_vector(
                        query_embedded,
                        partition_names=[comp],
                        limit=limit,
                    )
                )

            if rerankable and len(related_comps) > 1:
                yield t("reranking_docs", lang) + get_elapsed_tips(
                    start_time,
                    lang=lang,
                )
                total_docs = embeddings.rerank(query, total_docs)
                docs = total_docs[:10]
            else:
                docs = total_docs

    else:
        yield t("embedding_query", lang) + get_elapsed_tips(start_time, lang=lang)
        query_embedded = embeddings.embed_query(query)

        yield t("searching_docs", lang) + get_elapsed_tips(start_time, lang=lang)
        docs = doc_search_by_vector(
            query_embedded,
            limit=10,
        )

    yield t("llm_thinking", lang) + get_elapsed_tips(start_time, lang=lang)

    docs_content = "\n=====\n".join(
        [f"文档片段:\n\n" + chunk.page_content for i, chunk in enumerate(docs)]
    )

    if universal_rag:
        universal_prompt = (
            universal_rag_prompt if lang == "zh" else universal_rag_prompt_en
        )
        universal_rag_agent = AgentBase(prompt=universal_prompt, llm_model=llm_model)
        ans_itr = universal_rag_agent.stream(
            query, chat_history, document_snippets=docs_content
        )
    else:
        ans_itr = rag_agent.stream(query, chat_history, document_snippets=docs_content)

    visited = {}
    count = 0
    buffer: str = ""
    pruned_references = []
    get_first_token = False
    whole = ""
    for chunk in ans_itr:
        whole += chunk.content
        buffer += chunk.content
        if "[" in buffer and len(buffer) < 128:
            matches = re.findall(doc_cite_pattern, buffer)
            if len(matches) == 0:
                continue
            else:
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

    print("\n\n=== RAW Output ===\n\n" + whole, "\n\n===\n\n")

    if len(buffer) > 0:
        yield AIMessageChunk(content=buffer)

    if not show_refs:
        return

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

    for suffix in suffixes:
        yield AIMessageChunk(content=suffix + "\n")
