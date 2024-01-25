from __future__ import annotations

import logging
import os
import pathlib
import platform
from enum import Enum
from typing import Optional, Tuple

from langchain_core.env import get_runtime_environment
from langchain_core.pydantic_v1 import BaseModel

logger = logging.getLogger(__name__)

PLUGIN_VERSION = "0.1.0"
IP_INFO_URL = "https://ipinfo.io/ip"
CLASSIFIER_URL = os.getenv("PEBBLO_CLASSIFIER_URL", "http://localhost:8000/v1")

file_loader = [
    "JSONLoader",
    "S3FileLoader",
    "UnstructuredMarkdownLoader",
    "UnstructuredPDFLoader",
    "UnstructuredFileLoader",
    "UnstructuredJsonLoader",
    "PyPDFLoader",
    "GCSFileLoader",
    "AmazonTextractPDFLoader",
    "CSVLoader",
    "UnstructuredExcelLoader",
]

dir_loader = ["DirectoryLoader", "S3DirLoader", "PyPDFDirectoryLoader"]

in_memory = ["DataFrameLoader"]

LOADER_TYPE_MAPPING = {"file": file_loader, "dir": dir_loader, "in-memory": in_memory}

SUPPORTED_LOADERS = (*file_loader, *dir_loader, *in_memory)

logger = logging.getLogger(__name__)


class Environment(Enum):
    LOCAL = "local"


class Runtime(BaseModel):
    """
    OS, language details
    """

    type: Optional[str] = ""
    host: str
    path: str
    ip: Optional[str] = ""
    platform: str
    os: str
    os_version: str
    language: str
    language_version: str
    runtime: Optional[str] = ""


class Framework(BaseModel):
    """
    Langchain framework details
    """

    name: str
    version: str


class App(BaseModel):
    name: str
    owner: str
    description: Optional[str]
    load_id: str
    runtime: Runtime
    framework: Framework
    plugin_version: str


class Doc(BaseModel):
    """Per document details."""

    name: str
    owner: str
    docs: list
    plugin_version: str
    load_id: str
    loader_details: dict
    loading_end: bool
    source_owner: str


def get_full_path(path):
    if (
        not path
        or ("://" in path)
        or ("/" == path[0])
        or (path in ["unknown", "-", "in-memory"])
    ):
        return path
    full_path = pathlib.Path(path).resolve()
    return str(full_path)


def get_loader_type(loader: str):
    for loader_type, loaders in LOADER_TYPE_MAPPING.items():
        if loader in loaders:
            return loader_type
    return "unknown"


def get_loader_full_path(loader):
    from langchain_community.document_loaders import (
        DataFrameLoader,
        GCSFileLoader,
        S3FileLoader,
    )
    from langchain_community.document_loaders.base import BaseLoader

    location = "-"
    if not isinstance(loader, BaseLoader):
        logger.error(
            "loader is not derived from BaseLoader, source location will be unknown!"
        )
        return location
    loader_keys = loader.__dict__.keys()
    if "bucket" in loader_keys:
        if isinstance(loader, GCSFileLoader):
            location = f"gc://{loader.bucket}/{loader.blob}"
        elif isinstance(loader, S3FileLoader):
            location = f"s3://{loader.bucket}/{loader.key}"
    elif "path" in loader_keys:
        location = loader.path
    elif "file_path" in loader_keys:
        location = loader.file_path
    elif "web_paths" in loader_keys:
        location = loader.web_paths[0]
    # For in-memory types:
    elif isinstance(loader, DataFrameLoader):
        location = "in-memory"
    return get_full_path(str(location))


def get_runtime() -> Tuple[Framework, Runtime]:
    runtime_env = get_runtime_environment()
    framework = Framework(
        name="langchain", version=runtime_env.get("library_version", None)
    )
    uname = platform.uname()
    runtime = Runtime(
        host=uname.node,
        path=os.environ["PWD"],
        platform=runtime_env.get("platform", "unknown"),
        os=uname.system,
        os_version=uname.version,
        language=runtime_env.get("runtime", "unknown"),
        language_version=runtime_env.get("runtime_version", "unknown"),
    )

    if "Darwin" in runtime.os:
        runtime.type = "desktop"
        logger.debug("MacOS")
        local_runtime = get_local_runtime(Environment.LOCAL.value)
        runtime.ip = local_runtime.get("ip", "")
        runtime.runtime = local_runtime.get("runtime", Environment.LOCAL.value)
        return framework, runtime

    curr_runtime = get_local_runtime(Environment.LOCAL.value)

    runtime.type = curr_runtime.get("type", "unknown")
    runtime.ip = curr_runtime.get("ip", "")
    runtime.runtime = curr_runtime.get("runtime", "unknown")

    logger.debug(f"runtime {runtime}")
    logger.debug(f"framework {framework}")
    return framework, runtime


def get_local_runtime(service):
    """fetch local runtime."""
    import socket  # lazy imports

    import requests

    host = socket.gethostname()
    try:
        response = requests.get(IP_INFO_URL, timeout=2)
        if response.status_code == 200:
            public_ip = response.text
        else:
            logger.debug("public ip not found, setting localhost")
            public_ip = socket.gethostbyname(host)
    except Exception:
        logger.warning("Public IP not found, switching to localhost ip address.")
        try:
            public_ip = socket.gethostbyname(host)
        except Exception:
            public_ip = socket.gethostbyname("localhost")
    path = os.getcwd()
    name = host
    runtime = {
        "type": Environment.LOCAL.value,
        "host": host,
        "path": path,
        "ip": public_ip,
        "name": name,
        "runtime": service,
    }
    return runtime
