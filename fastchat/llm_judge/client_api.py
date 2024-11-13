import json
from typing import Any, List, Optional, Union
import requests
import time
import base64


class APIClient:
    def __init__(self, server_addr: str):
        self.completions_v1_url = f'{server_addr}/v1/chat/completions'
        self._models_v1_url = f'{server_addr}/v1/models'
        self.model_name = self.get_model_list(self._models_v1_url)[0]
        # print(self.model_name)

    @staticmethod
    def get_model_list(api_url: str):
        """Get model list from api server."""
        response = requests.get(api_url)
        if hasattr(response, 'text'):
            model_list = json.loads(response.text)
            model_list = model_list.pop('data', [])
            return [item['id'] for item in model_list]
        return None

    def v1_chat_completions(
            self,
            prompt: Union[str, List[Any]],
            temperature: Optional[float] = 0,
            max_tokens: Optional[int] = 2048,
            stream: Optional[bool] = False,
            top_p: Optional[float] = 1.0,
            top_k: Optional[int] = 1,
            repetition_penalty: Optional[float] = 1.0,
            **kwargs):
        pload = {
            'model': self.model_name,
            'messages': prompt,
            'stream': stream,
            'max_tokens': max_tokens,
            'top_k': top_k,
            'top_p': top_p,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty,
        }

        headers = {'content-type': 'application/json'}
        response = requests.post(self.completions_v1_url,
                                 headers=headers,
                                 json=pload,
                                 stream=stream)

        for chunk in response.iter_lines(chunk_size=8192,
                                         decode_unicode=False,
                                         delimiter=b'\n'):
            if chunk:
                if stream:
                    decoded = chunk.decode('utf-8')
                    if decoded == 'data: [DONE]':
                        continue
                    if decoded[:6] == 'data: ':
                        decoded = decoded[6:]
                    output = json.loads(decoded)
                    yield output
                else:
                    decoded = chunk.decode('utf-8')
                    output = json.loads(decoded)
                    yield output
