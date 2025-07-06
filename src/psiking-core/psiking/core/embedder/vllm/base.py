from typing import List, Optional

import httpx
from pydantic import BaseModel

from psiking.core.embedder.base import BaseEmbedder

class VLLMOnlineEmbedder(BaseEmbedder):
    def __init__(
        self,
        base_url: str,
        model: str,
        *args,
        api_key: str = "sk-abc",
        **kwargs
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        super().__init__(*args, **kwargs)
        
    def _post_tokenize(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_token_strs: bool = False,
    ):
        '''/tokenize API
        Request:
        {
            "model": "model-name",
            "prompt": "I am here",
            "add_special_tokens": true,
            "return_token_strs": false,
            "additionalProp1": {}
        }
        Response:
        {
            "count": 3,
            "max_model_len": 128000,
            "tokens": [40, 1079, 1588],
            "token_strs": ["I", "Ġam", "Ġhere"]
        }
        '''
        body = {
            'model': self.model,
            'prompt': text,
            'add_special_tokens': add_special_tokens,
            'return_token_strs': return_token_strs
        }
        
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/tokenize",
                headers={
                    'Authorization': f'Bearer {self.api_key}'
                },
                json=body,
            )
            response.raise_for_status()
            response_json = response.json()
        return response_json
    
    def _post_v1_embeddings(
        self,
        text: List[str],
        dimensions: Optional[int] = None,
        add_special_tokens: bool = False,
        encoding_format: str = "float",
    ):
        '''/v1/embeddings (OpenAI compatible) API
        Request:
        {
            "model": "model-name",
            "input": [0],
            "encoding_format": "float",
            "dimensions": 0,
            "user": "string",
            "truncate_prompt_tokens": -1,
            "additional_data": "string",
            "add_special_tokens": true,
            "priority": 0,
            "additionalProp1": {}
        }
        Response:
        {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [-0.1, -0.2,..],
                    "index": 0
                },
                {
                    "object": "embedding",
                    "embedding": [-0.1, -0.2,..],
                    "index": 1
                },
                ...
            ],
            "model": "model-name",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        } 
        '''
        body = {
            'model': self.model,
            'input': text,
            'add_special_tokens': add_special_tokens,
            'encoding_format': encoding_format
        }
        if dimensions is not None:
            body['dimensions'] = dimensions
        
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/v1/embeddings",
                headers={
                    'Authorization': f'Bearer {self.api_key}'
                },
                json=body,
            )
            response.raise_for_status()
            response_json = response.json()
        return response_json
    
    async def _apost_v1_embeddings(
        self,
        text: List[str],
        dimensions: Optional[int] = None,
        add_special_tokens: bool = False,
        encoding_format: str = "float",
    ):
        '''/v1/embeddings (OpenAI compatible) API async'''
        body = {
            'model': self.model,
            'input': text,
            'add_special_tokens': add_special_tokens,
            'encoding_format': encoding_format
        }
        if dimensions is not None:
            body['dimensions'] = dimensions
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                headers={
                    'Authorization': f'Bearer {self.api_key}'
                },
                json=body,
            )
            response.raise_for_status()
            response_json = response.json()
        return response_json
    
    def _post_pooling(
        self,
        texts: Optional[List[str]]=None,
        messages: Optional[List[dict]] = None,
        dimensions: Optional[int] = None,
        add_special_tokens: bool = False,
        encoding_format: str = "float",
        timeout: httpx.Timeout = httpx.Timeout(300.0)
    ):
        '''/pooling API
        Request:
        # Input Text only
        {
            "model": "model-name",
            "input": [0],
            "encoding_format": "float",
            "dimensions": 0,
            "user": "string",
            "truncate_prompt_tokens": -1,
            "additional_data": "string",
            "add_special_tokens": true,
            "priority": 0,
            "additionalProp1": {}
        }
        # Chat Template
        {
            "model": "model-name",
            "messages": [
                {"role": "user", "content": "..."}
            ]
        }
        
        Response:
        * embedding dim (1D or 2D) depends on model's pooler-config
        {
            "id": "pool-f91dd2b570d74fc1835a4085f4c101cd",
            "object": "list",
            "created": 1751630893,
            "model": "model-name",
            "data": [
                {
                    "index": 0,
                    "object": "pooling",
                    "data": [[-0.1, -0.2, ...], ..]
                },
                ...
            ],
            "usage": {"prompt_tokens": 4, "total_tokens": 4, "completion_tokens": 0, "prompt_tokens_details": null}
        }
        '''
        body = {
            'model': self.model,
            'add_special_tokens': add_special_tokens,
            'encoding_format': encoding_format
        }
        
        if not texts is None and not messages is None:
            raise ValueError("only one of texts or messages should be provided")
        elif texts is None and messages is None:
            raise ValueError("either one of texts or messages should be provided")
        elif texts is None:
            body['messages'] = messages
        else:
            body['input'] = texts
            
        if dimensions is not None:
            body['dimensions'] = dimensions
            
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/pooling",
                headers={
                    'Authorization': f'Bearer {self.api_key}'
                },
                json=body,
                timeout=timeout
            )
            response.raise_for_status()
            response_json = response.json()
        return response_json
    
    async def _apost_pooling(
        self,
        texts: Optional[List[str]]=None,
        messages: Optional[List[dict]]=None,
        dimensions: Optional[int] = None,
        add_special_tokens: bool = False,
        encoding_format: str = "float",
        timeout: httpx.Timeout = httpx.Timeout(300.0)
    ):
        '''/pooling API async'''
        body = {
            'model': self.model,
            'add_special_tokens': add_special_tokens,
            'encoding_format': encoding_format
        }
        
        if not texts is None and not messages is None:
            raise ValueError("only one of texts or messages should be provided")
        elif texts is None and messages is None:
            raise ValueError("either one of texts or messages should be provided")
        elif texts is None:
            body['messages'] = messages
        else:
            body['input'] = texts
        if dimensions is not None:
            body['dimensions'] = dimensions
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/pooling",
                headers={
                    'Authorization': f'Bearer {self.api_key}'
                },
                json=body,
                timeout=timeout
            )
            response.raise_for_status()
            response_json = response.json()
        return response_json