# jina-embeddings-v4-vllm-* Embedder

from typing import List, Literal, Union
from tqdm import tqdm

import numpy as np
from openai.types.chat import ChatCompletionMessageParam

from psiking.core.embedder.vllm.base import VLLMOnlineEmbedder

'''
calling with 'v1/embeddings' API may give following error
```
outputs=EmbeddingOutput.from_base(request_output.outputs),
...
raise ValueError("pooled_data should be a 1-D embedding vector")
```

-> recommends to use '/pooling' API
* https://github.com/vllm-project/vllm/issues/11446
* https://github.com/vllm-project/vllm/pull/11457

```
we should now use LLM.encode() for reward models
while LLM.embed() should be used for embedding models (not reward models)
'''

VISION_START_TOKEN_ID, VISION_END_TOKEN_ID = 151652, 151653

def pool_embedding(x: np.array):
    return np.sum(x, axis=0, dtype=np.float32) / x.shape[0]

def normalize_embedding(x: np.array):
    return x / np.linalg.norm(x, ord=2)

# def pool_and_normalize(x: np.array):
#     pooled_output = np.sum(x, axis=0, dtype=np.float32) / x.shape[0]
#     normalized_output = pooled_output / np.linalg.norm(pooled_output, ord=2)
#     return normalized_output

class VLLMOnlineJinaEmbV4Embedder(VLLMOnlineEmbedder):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    
    async def aembed_messages(
        self,
        input: List[dict]
    ):
        '''Returns /pooling result as is (2D list of shape (seq_len, hid_dim))'''
        response = await self._apost_pooling(messages=input)
        result = response['data'][0]['data']
        return result

    async def aembed_text(
        self,
        input: str,
        mode: Literal['query', 'passage'] = 'query',
    ) -> List[List[float]]:
        '''Returns /pooling result as is (2D list of shape (seq_len, hid_dim))'''
        prefix = "Query: " if mode=='query' else "Passage: "
        response = await self._apost_pooling(texts=[prefix+input])
        result = response['data'][0]['data']
        return result
        
    def embed_messages(
        self,
        input: List[dict]
    ):
        '''Returns /pooling result as is (2D list of shape (seq_len, hid_dim))'''
        response = self._post_pooling(messages=input)
        result = response['data'][0]['data']
        return result

    def embed_text(
        self,
        input: str,
        mode: Literal['query', 'passage'] = 'query',
    ) -> List[List[float]]:
        '''Returns /pooling result as is (2D list of shape (seq_len, hid_dim))'''
        prefix = "Query: " if mode=='query' else "Passage: "
        response = self._post_pooling(texts=[prefix+input])
        result = response['data'][0]['data']
        return result
        
    def run(
        self,
        input: Union[str, List[dict]],
        input_format: Literal['text', 'messages'] ='text',
        mode: Literal['query', 'passage'] = 'query',
        pool: bool = False,
        normalize: bool = False
    ) -> np.array:
        '''
        run only handles one text or messages
        Args
            input (Union[str, List[dict]]): Input text or messages
            input_format (str): indicate whether input should be treated as text or messages
            mode (str): query/passage
            pool (bool): whether to returned pooled embedding (hid_dim), returns (seq_len, hid_dim) otherwise
            normalize (bool): whether to normalize embedding
        '''
        if input_format=='text':
            embedding = self.embed_text(
                input=input,
                mode=mode
            )
        else:
            # Mesaages doesn't support mode
            embedding = self.embed_messages(
                input=input
            )
        embedding = np.array(embedding)
        if pool:
            embedding = pool_embedding(embedding)
        if normalize:
            embedding = normalize_embedding(embedding)
        return embedding
    
    async def arun(
        self,
        input: Union[str, List[dict]],
        input_format: Literal['text', 'messages'] ='text',
        mode: Literal['query', 'passage'] = 'query',
        pool: bool = False,
        normalize: bool = False
    ) -> np.array:
        '''
        run only handles one text or messages
        Args
            input (Union[str, List[dict]]): Input text or messages
            input_format (str): indicate whether input should be treated as text or messages
            mode (str): query/passage
            pool (bool): whether to returned pooled embedding (hid_dim), returns (seq_len, hid_dim) otherwise
            normalize (bool): whether to normalize embedding
        '''
        if input_format=='text':
            embedding = await self.aembed_text(
                input=input,
                mode=mode
            )
        else:
            # Mesaages doesn't support mode
            embedding = await self.aembed_messages(
                input=input
            )
        embedding = np.array(embedding)
        if pool:
            embedding = pool_embedding(embedding)
        if normalize:
            embedding = normalize_embedding(embedding)
        return embedding