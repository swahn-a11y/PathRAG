import json
import os
from typing import Callable, Any, Optional
import aiohttp
import numpy as np
import time
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .utils import (
    wrap_embedding_func_with_attrs,
)
from pydantic import BaseModel
from typing import List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tiktoken
import websocket
from websocket._exceptions import WebSocketTimeoutException

import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", None)
LLM_API_HEADER_KEY_NAME = os.getenv("LLM_API_HEADER_KEY_NAME", None)
LLM_API_HEADER_PW = os.getenv("LLM_API_HEADER_PW", None)

class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]

class WebSocketWithTimeout:
    """
    A WebSocket client with a timeout for connection and message receiving.
    """
    def __init__(self, url, headers, timeout=120):
        self.url = url
        self.timeout = timeout
        self.headers = headers
        self.ws = None

    def __enter__(self):
        self.ws = websocket.WebSocket()
        self.connection_with_timeout(self.timeout)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ws:
            self.ws.close()

    def connection_with_timeout(self, timeout):
        """
        Connect to the WebSocket server with a timeout.
        """
        self.ws.connect(self.url, header=self.headers)
        self.ws.sock.settimeout(timeout)

    def recv(self):
        """
        Receive a message from the WebSocket server with a timeout.
        """
        return self.ws.recv()


class WebSockerServerClosed(Exception):
    """
    Exception raised when the WebSocket server is closed unexpectedly.
    """


async def llm_call(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:

    messages = []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    model_params_for_gemini = {
        "system_instruction": system_prompt,
        "generation_config": {
            "temperature": 0.1,
            "top_p": 0.1,
            "max_output_tokens": 60000  # 50 page
        },
        "messages": messages,
    }

    data = {"from_where": "PathRAG", "model_name": "google-gemini-2.5-flash",
            "model_params": model_params_for_gemini}

    return await get_llm_response(data, **kwargs)


async def safe_requests_v2(url: str = None, method: str = 'get', **kwargs):
    """
    :param url: str
    :param method: str like get, post, patch
    :param kwargs: key: value arguments
    :return:
    """

    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        if method == 'get':
            response = session.get(url, **kwargs)
        elif method == 'post':
            response = session.post(url, **kwargs)
        else:
            raise ValueError("Unsupported HTTP method.")

        # Check if the response contains JSON
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        # if 'json' in response.headers.get('Content-Type'):
        #     return response.json()
        # else:
        #     return response.text
        return response
    except requests.exceptions.HTTPError as errh:
        return dict_to_dot({'status_code': 503, 'msg': f"HTTP Error: {errh}"})
    except requests.exceptions.ConnectionError as errc:
        return dict_to_dot({'status_code': 503, 'msg': f"Error Connecting: {errc}"})
    except requests.exceptions.Timeout as errt:
        return dict_to_dot({'status_code': 503, 'msg': f"Timeout Error: {errt}"})
    except requests.exceptions.RequestException as err:
        return dict_to_dot({'status_code': 503, 'msg': f"Request Error: {err}"})
    except Exception as e:
        print(e)

def waiting_llm_response(task_id: str, timeout=360):
    """
    Wait for the LLM response using WebSocket.
    """
    # send_slack_message(f"Waiting for LLM response for task_id: {task_id}")
    try:
        if "https://" in LLM_ENDPOINT:
            ws_endpoint = LLM_ENDPOINT.replace("https://", "")
            url = f"wss://{ws_endpoint}/waiting_llm_response/{task_id}"
        else:
            ws_endpoint = LLM_ENDPOINT.replace("http://", "")
            url = f"ws://{ws_endpoint}/waiting_llm_response/{task_id}"
        headers = {LLM_API_HEADER_KEY_NAME: LLM_API_HEADER_PW, "task_id": task_id}
        with WebSocketWithTimeout(url, headers, timeout=timeout) as ws:
            message = ws.recv()
            if not message:
                raise WebSockerServerClosed("WebSocket server closed the connection.")
            return json.loads(message)
    except Exception as e:
        return None


async def get_llm_response(data: dict, **kwargs):

    timeout = 360
    time_interval_sec = 2

    is_api = False

    max_token = 1039384
    
    def count_tokens(text: str, model: str = None):
        """

        Args:
            text: str
            model: str (optional. if not specified, model name will be 'cl100k_base')

        Returns: count: int

        """
        if model:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except:
                encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        encoded_tokens = encoding.encode(text)
        return len(encoded_tokens)

    def truncate_message(message: dict, cut_number: int):
        # cut_number has already buffer, so calculation here is not strict.
        if "content" not in message:
            raise ValueError("message['content'] is not valid @util.get_llm_response")
        token_count_message = count_tokens(json.dumps(message, ensure_ascii=False))
        # print(f"token_count_message: {token_count_message}")
        if isinstance(message["content"], str):
            token_count_message_content = count_tokens(message["content"])
            token_count_excluding_content = token_count_message - token_count_message_content
            message['content'] = message['content'][:cut_number - token_count_excluding_content]
        elif isinstance(message["content"], list):
            token_count_message_content = count_tokens(json.dumps(message["content"], ensure_ascii=False))
            token_count_excluding_content = token_count_message - token_count_message_content
            if message['content'][-1]["type"] == "text":
                message['content'][-1]["text"] = message['content'][-1]["text"][:cut_number - token_count_excluding_content]
            elif message['content'][-1]["type"]  == "tool_result":
                message['content'][-1]["content"] = message['content'][-1]["content"][:cut_number - token_count_excluding_content]
            else:
                raise ValueError("message['content'] is not valid @util.get_llm_response")
        else:
            raise ValueError("message['content'] is not valid @util.get_llm_response")
        return message


    if not isinstance(data, dict) or "from_where" not in data or "model_name" not in data or "model_params" not in data:
        raise ValueError("data is not valid!")

    token_count_model_params = count_tokens(json.dumps(data['model_params'], ensure_ascii=False))
    if token_count_model_params > max_token:
        messages = data['model_params']['messages']
        token_count_messages = count_tokens(json.dumps(messages, ensure_ascii=False))
        token_count_excluding_messages = token_count_model_params - token_count_messages

        if messages[0]["role"].lower() == "system":
            system_message_in_messages = messages.pop(0)   # in case of openai
            token_count_system_message = count_tokens(json.dumps(system_message_in_messages, ensure_ascii=False))
        else:
            system_message_in_messages = None
            token_count_system_message = 0
        while token_count_messages > max_token - token_count_excluding_messages - token_count_system_message:
            if len(messages) == 1:
                cut_number = token_count_messages - (max_token - token_count_excluding_messages - token_count_system_message) - 500  # 500 -> buffer
                if cut_number >= 500:
                    messages[0] = truncate_message(messages[0], cut_number)
                else:
                    raise ValueError(f"data['model_params'] exceeds max token: {max_token} of {data['model_name']} @util.get_llm_response")
                break
            messages.pop(0)
            token_count_messages = count_tokens(json.dumps(messages, ensure_ascii=False))
        messages = [system_message_in_messages] + messages if system_message_in_messages else messages
        data['model_params']['messages'] = messages

    headers = {'Content-Type': 'application/json; charset=utf-8', LLM_API_HEADER_KEY_NAME: LLM_API_HEADER_PW}

    response = await safe_requests_v2(f"{LLM_ENDPOINT}/call_llm_websocket/", 'post', json=data,
                                 params={"is_api": is_api}, headers=headers, timeout=10)

    task_id = None
    status_code = response.status_code
    if status_code == 202:
        task_id = response.json()['task_id']

        time.sleep(time_interval_sec)
    elif status_code == 200:
        status_data = response.json()
        if status_data["status"] == "completed":
            result = json.loads(status_data['result'])
            return result["candidates"][0]["content"]["parts"][0]["text"]
    elif status_code == 400:
        err_msg = f"[get_llm_response@utils] Value Error - status_code: {status_code}"
        raise ValueError(err_msg)
    else:
        err_msg = f"[get_llm_response@utils] System Error - status_code: {status_code}"
        raise SystemError(err_msg)

    status_data = waiting_llm_response(task_id, timeout=timeout)

    if status_data is None:
        return None

    result = status_data['result']

    result = json.loads(result)

    return result["candidates"][0]["content"]["parts"][0]["text"]

@wrap_embedding_func_with_attrs(embedding_dim=3072, max_token_size=8191)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_embedding(
    texts: list[str],
    model: str = "ailab-embedding-3-large",
    base_url=os.getenv("AZURE_EMBEDDING_TEXT3_API_ENDPOINT"),
    api_key=os.getenv("AZURE_EMBEDDING_TEXT3_API_KEY"),
    api_version: str = "2023-05-15",
) -> np.ndarray:

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=base_url,
        api_key=api_key,
        api_version=api_version,
    )

    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

