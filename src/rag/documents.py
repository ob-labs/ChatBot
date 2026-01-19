import os
import re
import tqdm
from pydantic import BaseModel
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from typing import Iterator


class DocumentMeta(BaseModel):
    """
    Document metadata.
    """

    class Config:
        extra = "allow"

    doc_url: str
    doc_name: str
    component: str = "observer"  # default component
    chunk_title: str
    enhanced_title: str


component_mapping = {
    "observer": 1,
    "ocp": 2,
    "oms": 3,
    "obd": 4,
    "operator": 5,
    "odp": 6,
    "odc": 7,
}


headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
    ("####", "Header4"),
    ("#####", "Header5"),
    ("######", "Header6"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
)


def parse_md(
    file_path: str,
    max_chunk_size: int = 4096,
) -> Iterator[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
        chunks = splitter.split_text(file_content)
        for chunk in chunks:
            subtitles = list(chunk.metadata.values())
            if len(subtitles) == 0:
                subtitles.append(file_path.split("/")[-1])
            meta = DocumentMeta(
                doc_url=file_path,
                chunk_title=subtitles[-1],
                enhanced_title=" -> ".join(subtitles),
                doc_name=chunk.metadata.get("Header1", subtitles[-1]),
            )
            if len(chunk.page_content) <= max_chunk_size:
                chunk.metadata = meta.model_dump()
                yield chunk
            else:
                for i in range(0, len(chunk.page_content), max_chunk_size):
                    sub_chunk = Document(chunk.page_content[i : i + max_chunk_size])
                    sub_chunk.metadata = meta.model_dump()
                    yield sub_chunk


class MarkdownDocumentsLoader:
    """
    Markdown Documents Loader.
    """

    def __init__(
        self,
        doc_base: str,
        skip_patterns: list[str] = [],
    ):
        self.doc_base = doc_base
        self.skip_patterns = skip_patterns

    def load(
        self,
        show_progress: bool = True,
        limit: int = 0,
        max_chunk_size: int = 4096,
    ) -> Iterator[Document]:
        """
        Load Markdown Documents from doc_base.
        Returns:
            Iterator of Document with DocumentMeta.
        """
        files_to_process: list[str] = []
        for root, _, files in os.walk(self.doc_base):
            for file in files:
                if file.endswith(".md") or file.endswith(".mdx"):
                    file_path = os.path.join(root, file)
                    if any(re.search(regex, file_path) for regex in self.skip_patterns):
                        continue
                    files_to_process.append(file_path)

        progress_wrapper = tqdm.tqdm if show_progress else lambda x: x
        files_to_process: list[str] = progress_wrapper(files_to_process)

        count = 0
        for file_path in files_to_process:
            for chunk in parse_md(file_path, max_chunk_size=max_chunk_size):
                yield chunk
            count += 1
            if limit > 0 and count >= limit:
                print(f"Limit reached: {limit}, exiting.")
                exit(0)
