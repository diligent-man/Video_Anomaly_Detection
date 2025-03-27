from typing import Type

import requests
import requests as rq
from requests import Timeout

from urllib3.util import Retry
from requests.adapters import HTTPAdapter


__all__ = ["ping_server"]


def ping_server(uri: str, **kwargs) -> int | Type[Timeout]:
    try:
        with requests.Session() as s:
            retries = Retry(
                total=kwargs.pop("total", 3),
                backoff_factor=0.5,
                allowed_methods=["GET"]
            )
            s.mount(uri, HTTPAdapter(max_retries=retries))
            response = s.get(uri, **kwargs)
        return response.status_code
    except rq.exceptions.Timeout:
        # Catching both ConnectTimeout and ReadTimeout errors
        return rq.exceptions.Timeout
