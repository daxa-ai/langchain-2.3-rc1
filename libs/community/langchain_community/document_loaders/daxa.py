"""Daxa's safe loader."""

import os
import requests
from http import HTTPStatus
import logging

from langchain_community.utilities.daxa import CLASSIFIER_URL, PLUGIN_VERSION
from langchain_community.utilities.daxa import get_loader_full_path, get_loader_type, get_full_path, get_runtime
from langchain_community.utilities.daxa import App, Framework, Runtime

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class DaxaSafeLoader(BaseLoader):

    def __init__(self, langchain_loader: BaseLoader, app_id: str):
        if not app_id or not isinstance(app_id, str):
            raise NameError(
                """No app_id provided. Or invalid app_id.
                NOTE: Make sure to use same app_id while using
                DaxaSafeLoader and DaxaCallbackHandler.from_credentials().
                """
            )
        self.app_name = app_id
        self.loader = langchain_loader
        self.source_path = get_loader_full_path(self.loader)
        self.docs = []
        loader_name = str(type(self.loader)).split(".")[-1].split("'")[0]
        source_type = get_loader_type(loader_name)
        self.loader_details = {
            "loader": loader_name,
            "source_path": self.source_path,
            "source_type": source_type
        }
        #generate app
        app = self._get_app_details()
        self._send_discover(app)

    def load(self):
        """load Documents."""
        self.docs = self.loader.load()
        self._send_loader_doc()
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

    def _send_loader_doc(self):
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        doc_content = [doc.dict() for doc in self.docs]
        payload = {
            "name": self.app_name,
            "docs": [{"doc": doc.get('page_content'), "source_path": get_full_path(doc.get('metadata', {}).get('source')), "last_modified": doc.get('metadata', {}).get('last_modified')} for doc in doc_content],
            "plugin_version": PLUGIN_VERSION,
            "loader_details": self.loader_details,
        }
        resp = requests.post(f"{CLASSIFIER_URL}/loader/doc", headers=headers, json=payload, timeout=10)
        logger.info(f"===> send_loader_doc: request, url {resp.request.url}, headers {resp.request.headers}, body {resp.request.body[:999]} with a len: {len(resp.request.body)}\n")
        logger.info(f"===> send_loader_doc: response status {resp.status_code}, body {resp.json()}\n")

    def _send_discover(self, app: App):
        headers =  {'Accept': 'application/json', 'Content-Type': 'application/json'}
        payload = app.dict(skip_defaults=True)
        resp = requests.post(f"{self.context.classifier_url}/discover", headers=headers, json=payload)
        logger.info(f"===> send_discover: request, url {resp.request.url}, headers {resp.request.headers}, body {resp.request.body}\n")
        logger.info(f"===> send_discover: response status {resp.status_code}, body {resp.json()}\n")
        if resp.status_code == HTTPStatus.OK or resp.status_code == HTTPStatus.BAD_GATEWAY:
            DaxaCallbackHandler.set_discover_sent()

    def _get_app_details(self):
        framework, runtime, cloud = get_runtime()
        app = App(
            name=self.app_name,
            runtime=runtime,
            framework=framework,
            plugin_version=PLUGIN_VERSION,
                )
        return app