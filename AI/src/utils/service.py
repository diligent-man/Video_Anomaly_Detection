import requests
import requests as rq

from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from typing import Tuple


__all__ = ["ping_server"]


def ping_server(uri: str, **kwargs) -> int:
    print(kwargs)
    try:
        with requests.Session() as s:
            retries = Retry(
                total=kwargs.pop("total", 5),
                backoff_factor=0.5,
                allowed_methods=["GET"]
            )
            s.mount(uri, HTTPAdapter(max_retries=retries))
            response = s.get(uri, **kwargs)
        return response.status_code
    except rq.exceptions.Timeout:
        # Catching both ConnectTimeout and ReadTimeout errors
        return 404
