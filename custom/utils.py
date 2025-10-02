from dataclasses import dataclass
import asyncio
import numpy as np
import logging
from functools import wraps

logger = logging.getLogger("PathRAG")

class UnlimitedSemaphore:
    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable
    concurrent_limit: int = 16

    def __post_init__(self):
        if self.concurrent_limit != 0:
            self._semaphore = asyncio.Semaphore(self.concurrent_limit)
        else:
            self._semaphore = UnlimitedSemaphore()

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        async with self._semaphore:
            return await self.func(*args, **kwargs)

def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):

    def final_decro(func):

        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro

def wrap_embedding_func_with_attrs(**kwargs):

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro