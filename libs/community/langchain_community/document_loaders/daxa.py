"""Daxa's safe loader."""

import os
import pwd
import requests
import logging
import uuid
from http import HTTPStatus

from langchain_community.utilities.daxa import CLASSIFIER_URL, PLUGIN_VERSION
from langchain_community.utilities.daxa import get_loader_full_path, get_loader_type, get_full_path, get_runtime
from langchain_community.utilities.daxa import App, Doc
from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class DaxaSafeLoader(BaseLoader):

    def __init__(self, langchain_loader: BaseLoader, app_id: str, owner: str, description: str=""):
        if not app_id or not isinstance(app_id, str):
            raise NameError("""No app_id is passed or invalid app_id.""")
        if not owner or not isinstance(owner, str):
            raise NameError("""No owner is passed or invalid owner.""")
        self.app_name = app_id
        self.load_id = str(uuid.uuid4())
        self.loader = langchain_loader
        self.owner = owner
        self.description = description
        self.source_path = get_loader_full_path(self.loader)
        self.source_owner = DaxaSafeLoader.get_file_owner_from_path(self.source_path)
        self.docs = []
        loader_name = str(type(self.loader)).split(".")[-1].split("'")[0]
        self.source_type = get_loader_type(loader_name)
        self.source_size = self.get_source_size(self.source_path)
        self.loader_details = {
            "loader": loader_name,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "source_size": self.source_size,
        }
        #generate app
        self.app = self._get_app_details()
        self._send_discover()

    def load(self):
        """load Documents."""
        self.docs = self.loader.load()
        self._send_loader_doc(loading_end=True)
        DaxaSafeLoader.set_loader_sent()
        return self.docs

    def lazy_load(self):
        """Lazy load Documents."""
        try:
            doc_iterator = self.loader.lazy_load()
        except NotImplementedError as exc:
            err_str = f"{self.__class__.__name__} does not implement lazy_load()"
            logger.error(err_str)
            raise NotImplementedError(err_str) from exc
        while True:
            try:
                doc = next(doc_iterator)
            except StopIteration:
                self.docs = [ ]
                self._send_loader_doc(loading_end=True)
                DaxaSafeLoader.set_loader_sent()
                break
            self.docs = [doc, ]
            self._send_loader_doc()
            yield self.docs

    @classmethod
    def set_discover_sent(cls):
        cls._discover_sent = True

    @classmethod
    def set_loader_sent(cls):
        cls._loader_sent = True

    def _send_loader_doc(self, loading_end=False):
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        doc_content = [doc.dict() for doc in self.docs]
        docs = []
        for doc in doc_content:
            doc_source_path = get_full_path(doc.get('metadata', {}).get('source'))
            doc_source_owner = DaxaSafeLoader.get_file_owner_from_path(doc_source_path)
            doc_source_size = self.get_source_size(doc_source_path)
            docs.append({"doc": doc.get('page_content'), "source_path": doc_source_path, "last_modified": doc.get('metadata', {}).get('last_modified'), "file_owner": doc_source_owner, "source_size": doc_source_size })
        payload = {
            "name": self.app_name,
            "owner": self.owner,
            "docs": docs,
            "plugin_version": PLUGIN_VERSION,
            "load_id": self.load_id,
            "loader_details": self.loader_details,
            "loading_end": "false",
            "file_owner": self.source_owner
        }
        if loading_end is True:
            payload["loading_end"] = "true"
        payload = Doc.model_validate(payload).model_dump(exclude_unset=True)
        resp = requests.post(f"{CLASSIFIER_URL}/loader/doc", headers=headers, json=payload, timeout=20)
        logger.debug(f"===> send_loader_doc: request, url {resp.request.url}, headers {resp.request.headers}, body {resp.request.body[:999]} with a len: {len(resp.request.body)}\n")
        logger.debug(f"===> send_loader_doc: response status {resp.status_code}, body {resp.json()}\n")

    def _send_discover(self):
        headers =  {'Accept': 'application/json', 'Content-Type': 'application/json'}
        payload = self.app.model_dump(exclude_unset=True)
        resp = requests.post(f"{CLASSIFIER_URL}/app/discover", headers=headers, json=payload, timeout=20)
        logger.debug(f"===> send_discover: request, url {resp.request.url}, headers {resp.request.headers}, body {resp.request.body}\n")
        logger.debug(f"===> send_discover: response status {resp.status_code}, body {resp.json()}\n")
        if resp.status_code == HTTPStatus.OK or resp.status_code == HTTPStatus.BAD_GATEWAY:
            DaxaSafeLoader.set_discover_sent()

    def _get_app_details(self):
        framework, runtime = get_runtime()
        app = App(
            name=self.app_name,
            owner=self.owner,
            description=self.description,
            load_id=self.load_id,
            runtime=runtime,
            framework=framework,
            plugin_version=PLUGIN_VERSION,
                )
        return app

    @staticmethod
    def get_file_owner_from_path(file_path: str) -> str:
        try:
            file_owner_uid=os.stat(file_path).st_uid
            file_owner_name = pwd.getpwuid(file_owner_uid).pw_name
        except Exception:
            file_owner_name = 'unknown'
        return file_owner_name

    def get_source_size(self, source_path: str) -> int:
        if os.path.isfile(source_path):
            size = os.path.getsize(source_path)
        elif os.path.isdir(source_path):
            total_size = 0
            for dirpath, _, filenames in os.walk(source_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
            size = total_size
        return size
