import json
from typing import Any, List, Optional, Union
import requests
import time
import base64


class APIClient:
    def __init__(self, server_addr: str):
        self.completions_v1_url = f"{server_addr}/v1/chat/completions"
        self._models_v1_url = f"{server_addr}/v1/models"
        self.model_name = self.get_model_list(self._models_v1_url)[0]
        # print(self.model_name)

    @staticmethod
    def get_model_list(api_url: str):
        """Get model list from api server."""
        response = requests.get(api_url)
        if hasattr(response, "text"):
            model_list = json.loads(response.text)
            model_list = model_list.pop("data", [])
            return [item["id"] for item in model_list]
        return None

    def v1_chat_completions(
        self,
        prompt: Union[str, List[Any]],
        temperature: Optional[float] = 0,
        max_tokens: Optional[int] = 2048,
        stream: Optional[bool] = False,
        top_p: Optional[float] = 0.8,
        top_k: Optional[int] = 10,
        repetition_penalty: Optional[float] = 1.06,
        **kwargs,
    ):
        pload = {
            "model": self.model_name,
            "messages": prompt,
            "stream": stream,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        }

        headers = {"content-type": "application/json"}
        response = requests.post(
            self.completions_v1_url, headers=headers, json=pload, stream=stream
        )

        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"\n"
        ):
            if chunk:
                if stream:
                    decoded = chunk.decode("utf-8")
                    if decoded == "data: [DONE]":
                        continue
                    if decoded[:6] == "data: ":
                        decoded = decoded[6:]
                    output = json.loads(decoded)
                    yield output
                else:
                    decoded = chunk.decode("utf-8")
                    output = json.loads(decoded)
                    yield output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Send a request to API with custom text input.")
    parser.add_argument(
        "--text-input",
        type=str,
        required=True,
        help="The text content for the prompt. Example: 'Draft a professional email seeking your supervisor's feedback on the Quarterly Financial Report.'"
    )
    

    server_addr = 'http://0.0.0.0:8080'
    api_client = APIClient(server_addr)
    # image_url = 'https://cf.shopee.co.id/file/id-11134207-7r98u-lv1eywfgw2uh74'
    # folder = '/home/jian.yin/code/llm/downloads/downloads/vlmdata/textvqa/llava_textvqa_val_v051_ocr.jsonl'
    image_url: str = '/workspace/code/downloads/downloads/vlmdata/textvqa/train_images/003a8ae2ef43b901.jpg'
    # base64_image = encode_image(image_url)
    # image_url = f"data:image/jpeg;base64,{base64_image}"
    # print(image_url[0])
    args = parser.parse_args()

    # 通过命令行参数获取 text_input
    text_input = args.text_input
    print(text_input)
    prompts = [{
        'role': 'user',
        'content': text_input,
    }]

    stream = False
    top_p = 1.0
    top_k = 1
    temperature = 0
    if stream:
        for result in api_client.v1_chat_completions(
                prompt=prompts,
                stream=stream,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
        ):
            text = result['choices'][0].get('delta').get('content')
            if text:
                print(text, end='', flush=True)
        print("\n")
    else:
        begin_time = time.time()
        for output in api_client.v1_chat_completions(
                prompt=prompts,
                stream=stream,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
        ):
            print(output)
        lantency = time.time() - begin_time
        print("lantency:", lantency)

