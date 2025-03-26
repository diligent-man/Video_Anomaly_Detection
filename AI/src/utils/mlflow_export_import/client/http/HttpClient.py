import os
import json
from typing import Dict, Any, Callable

import requests as rq
from requests import Response
from overrides import override

from .BaseHttpClient import BaseHttpClient
from ...common import ExportImportException


__all__ = ["HttpClient"]


class HttpClient(BaseHttpClient):
    """
    Wrapper for HTTP calls for MLflow Databricks APIs.
    """
    __USER_AGENT: str = "mlflow-export-import/1.0.0"
    __TIMEOUT: int = 120  # per mlflow.MlflowClient

    def __init__(self, api_name: str, host: str = None, token: str = None) -> None:
        """TIMEOUT
        :param api_name: Name of base API such as 'api/2.0' or 'api/2.0/mlflow'.
        :param host: Host name of tracking server (e.g. 'http://localhost:5000').
        :param token: Databricks token if using Databricks.
        """
        if host is None:
            raise MlflowExportImportException(
                "MLflow tracking URI (MLFLOW_TRACKING_URI environment variable) is not configured correctly",
                http_status_code=401
            )

        self.host: str = host
        self.token: str = token
        self.api_uri: str = os.path.join(host, api_name)

    @override
    def __repr__(self) -> str:
        return self.api_uri
    ####################################################################################################################

    @override
    def _get(self, resource: Any, params=None) -> Response:
        uri: str = self._mk_uri(resource)
        rsp: Response = rq.get(
            uri,
            headers=self._mk_headers(),
            data=params,
            timeout=self.__TIMEOUT
        )
        return self._check_response(rsp, params)

    @override
    def _put(self, resource: Any, data=None) -> Response:
        return self._mutator(rq.put, resource, data)

    @override
    def _post(self, resource: Any, data=None) -> Response:
        return self._mutator(rq.post, resource, data)

    @override
    def _patch(self, resource: Any, data=None) -> Response:
        return self._mutator(rq.patch, resource, data)

    @override
    def _delete(self, resource: Any) -> Response:
        uri: str = self._mk_uri(resource)
        rsp: Response = rq.delete(
            uri,
            headers=self._mk_headers(),
            timeout=self.__TIMEOUT
        )
        return self._check_response(rsp)
    ####################################################################################################################

    @staticmethod
    def _json_dumps(data: Any) -> None | str:
        return json.dumps(data) if data else None

    @staticmethod
    def _get_response_text(rsp: Response) -> Any:
        try:
            return rsp.json()
        except rq.exceptions.JSONDecodeError:
            return rsp.text

    @staticmethod
    def _check_response(rsp: Response, params=None) -> Response:
        """
        :param rsp: returned Response object
        :param params: passed paras
        :return: checked Response object
        """
        if rsp.status_code < 200 or rsp.status_code > 299:
            msg = json.dumps({
                "http_status_code": rsp.status_code,
                "uri": rsp.url,
                "params": params,
                "response": rsp.text
            }, indent=4)
            raise MlflowExportImportException(msg, http_status_code=rsp.status_code)
        return rsp

    @staticmethod
    def _json_loads(rsp: Response, params) -> Any:
        json_str = rsp.text
        try:
            return json.loads(json_str)
        except json.decoder.JSONDecodeError as e:
            import traceback
            traceback.print_exc()
            msg = {
                "uri": rsp.url,
                "method": rsp.request.method,
                "params": params,
                "exception": str(e),
                "response": json_str
            }
            raise MlflowExportImportException(msg, http_status_code=rsp.status_code)

    def _mutator(self, method: Callable, resource: Any, data=None) -> Response:
        """
        :param method: Can be PUT/ POST/ PATCH http request
        :param resource: resource to send
        :param data: data to send
        :return:
        """
        uri: str = self._mk_uri(resource)
        rsp: Response = method(
            uri,
            headers=self._mk_headers(),
            data=data,
            timeout=self.__TIMEOUT
        )
        return self._check_response(rsp)

    def _mk_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "User-Agent": self.__USER_AGENT,
            "Content-Type": "application/json"
        }

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _mk_uri(self, resource: Any) -> str:
        return f"{self.api_uri}/{resource}"
    ####################################################################################################################

    @override
    def get(self, resource: Any, params=None) -> Any:
        """
        :param resource: Relative path name of resource such as experiments/search
        :param params: Dict of query parameters
        """
        rsp: Response = self._get(resource, self._json_dumps(params))
        return self._json_loads(rsp, params)

    @override
    def post(self, resource: Any, data=None) -> Any:
        """
        :param resource: Relative path name of resource such as runs/search
        :param data: Request payload as dict
        """
        rsp: Response = self._post(resource, self._json_dumps(data))
        return self._json_loads(rsp, data)

    @override
    def put(self, resource: Any, data=None) -> Any:
        """
        :param resource: Relative path name of resource
        :param data: Request payload as dict
        """
        rsp: Response = self._put(resource, self._json_dumps(data))
        return self._json_loads(rsp, data)

    @override
    def patch(self, resource: Any, data=None) -> Any:
        """
        :param resource: Relative path name of resource
        :param data: Request payload as dict
        """
        rsp: Response = self._patch(resource, self._json_dumps(data))
        return self._json_loads(rsp, data)

    @override
    def delete(self, resource: Any) -> Any:
        """
        :param resource: Relative path name of resource such as runs/search
        """
        return json.loads(self._delete(resource).text)

    def get_api_uri(self) -> str:
        return self.api_uri

    def get_token(self) -> str:
        return self.token
