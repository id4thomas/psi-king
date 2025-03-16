from typing import Optional, Union
from urllib.parse import urljoin

import httpx
import backoff

def is_status_400_error(e: Exception) -> bool:
    if isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500:
        return True
    else:
        return False
    
class HTTPXClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    @backoff.on_exception(
        backoff.expo,
        (
            httpx.ConnectTimeout,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.HTTPStatusError,
        ),
        max_tries=4,
        jitter=backoff.full_jitter,
        giveup=is_status_400_error,
    )
    def get(
        self,
        path: str,
        params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = httpx.Timeout(5.0)
    ) -> httpx.Response:
        url = urljoin(self.base_url, path)
        headers = {"Content-Type": "application/json"}
        with httpx.Client() as client:
            response = client.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response
    
    @backoff.on_exception(
        backoff.expo,
        (
            httpx.ConnectTimeout,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.HTTPStatusError,
        ),
        max_tries=4,
        jitter=backoff.full_jitter,
        giveup=is_status_400_error,
    )
    async def aget(
        self,
        path: str,
        params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = httpx.Timeout(5.0)
    ) -> httpx.Response:
        url = urljoin(self.base_url, path)
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response

    @backoff.on_exception(
        backoff.expo,
        (httpx.ConnectTimeout, httpx.ConnectError, httpx.NetworkError),
        max_tries=4,
        jitter=backoff.full_jitter,
    )
    def post(
        self, path: str, data: Optional[dict] = None, timeout: Optional[Union[float, httpx.Timeout]] = httpx.Timeout(5.0)
    ) -> httpx.Response:
        url = urljoin(self.base_url, path)
        headers = {"Content-Type": "application/json"}
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response

    @backoff.on_exception(
        backoff.expo,
        (httpx.ConnectTimeout, httpx.ConnectError, httpx.NetworkError),
        max_tries=4,
        jitter=backoff.full_jitter,
    )
    async def apost(
        self, path: str, data: Optional[dict] = None, timeout: Optional[Union[float, httpx.Timeout]] = httpx.Timeout(5.0)
    ) -> httpx.Response:
        url = urljoin(self.base_url, path)
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response

    @backoff.on_exception(
        backoff.expo,
        (httpx.ConnectTimeout, httpx.ConnectError, httpx.NetworkError),
        max_tries=4,
        jitter=backoff.full_jitter,
    )
    def delete(self, path: str, params: Optional[dict] = None, data: Optional[dict] = None) -> httpx.Response:
        url = urljoin(self.base_url, path)
        headers = {"Content-Type": "application/json"}
        with httpx.Client() as client:
            if data:
                ## delete method doesn't support json arg
                # https://github.com/encode/httpx/discussions/1587
                response = client.request(
                    method="DELETE", url=url, params=params, headers=headers, json=data
                )
            else:
                response = client.delete(url, params=params, headers=headers)

        response.raise_for_status()
        return response
    
    @backoff.on_exception(
        backoff.expo,
        (httpx.ConnectTimeout, httpx.ConnectError, httpx.NetworkError),
        max_tries=4,
        jitter=backoff.full_jitter,
    )
    async def adelete(self, path: str, params: Optional[dict] = None, data: Optional[dict] = None) -> httpx.Response:
        url = urljoin(self.base_url, path)
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            if data:
                ## delete method doesn't support json arg
                # https://github.com/encode/httpx/discussions/1587
                response = await client.request(
                    method="DELETE", url=url, params=params, headers=headers, json=data
                )
            else:
                response = await client.delete(url, params=params, headers=headers)

        response.raise_for_status()
        return response

    @backoff.on_exception(
        backoff.expo,
        (httpx.ConnectTimeout, httpx.ConnectError, httpx.NetworkError),
        max_tries=4,
        jitter=backoff.full_jitter,
    )
    def put(self, path: str, data: Optional[dict] = None) -> httpx.Response:
        url = urljoin(self.base_url, path)
        headers = {"Content-Type": "application/json"}
        with httpx.Client() as client:
            response = client.put(url, json=data, headers=headers)
        response.raise_for_status()
        if response.status_code == 204:
            return None
        return response
    
    @backoff.on_exception(
        backoff.expo,
        (httpx.ConnectTimeout, httpx.ConnectError, httpx.NetworkError),
        max_tries=4,
        jitter=backoff.full_jitter,
    )
    async def aput(self, path: str, data: Optional[dict] = None) -> httpx.Response:
        url = urljoin(self.base_url, path)
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            response = await client.put(url, json=data, headers=headers)
        response.raise_for_status()
        if response.status_code == 204:
            return None
        return response