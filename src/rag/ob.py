"""
OceanBase specific utilities for document RAG.

This module contains OceanBase-specific components including:
- Vector store initialization and configuration
- Document search functions using OceanBase vector store
- Component mapping and URL replacement utilities
"""

import re
from typing import List, Optional

from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from langchain_core.documents import Document
from pyobvector import RangeListPartInfo
from sqlalchemy import Column, Integer

from src.common.config import (
    EmbeddingConfig,
    get_echo,
    get_table_name,
)
from src.common.db import ConnectionParams
from src.common.logger import get_logger
from src.rag.embedding import get_embedding

logger = get_logger(__name__)

# Configuration constants
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_RERANK_LIMIT = 10
DEFAULT_COMPONENT = "observer"

# Component name to code mapping
component_mapping = {
    "default_modules": 0,
    "observer": 1,
    "ocp": 2,
    "oms": 3,
    "obd": 4,
    "operator": 5,
    "odp": 6,
    "odc": 7,
}

# Supported components
supported_components = component_mapping.keys()

# URL replacement patterns for documentation links
URL_REPLACERS = [
    (r"^.*oceanbase-doc", "https://github.com/oceanbase/oceanbase-doc/blob/V4.3.5"),
    (r"^.*ocp-doc", "https://github.com/oceanbase/ocp-doc/blob/V4.4.0"),
    (r"^.*odc-doc", "https://github.com/oceanbase/odc-doc/blob/V4.3.3"),
    (r"^.*oms-doc", "https://github.com/oceanbase/oms-doc/tree/V4.2.11"),
    (r"^.*obd-doc", "https://github.com/oceanbase/obd-doc/tree/V4.0.0"),
    (
        r"^.*oceanbase-proxy-doc",
        "https://github.com/oceanbase/oceanbase-proxy-doc/blob/V4.3.5",
    ),
    (r"^.*?ob-operator/", "https://github.com/oceanbase/ob-operator/blob/master/"),
]

def get_part_list() -> List[RangeListPartInfo]:
    """
    Get partition list for OceanBase vector store.

    Returns:
        List of RangeListPartInfo objects including all component mappings
        and a default partition.
    """
    # Filter out default_modules from component_mapping because it needs special handling
    # (uses "DEFAULT" string instead of numeric value 0)
    return [
        RangeListPartInfo(k, v) for k, v in component_mapping.items() if k != "default_modules"
    ] + [RangeListPartInfo("default_modules", "DEFAULT")]


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
