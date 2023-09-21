import os
from typing import List, Optional
from langchain.llms.base import LLM
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
import torch
path="/root/huggingface/hub/models--QWen--QWen-7B-Chat/snapshots/396efbeb1e47da72b50878978a9b366fd9db0114"
# initialize qwen 7B model
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, fp16=True).eval()
model.generation_config = GenerationConfig.from_pretrained(path, trust_remote_code=True) 


# torch garbage collection
def torch_gc():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    DEVICE = "cuda"
    DEVICE_ID = "0"
    CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
    a = torch.Tensor([1, 2])
    a = a.cuda()
    print(a)

    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# wrap the qwen model with langchain LLM base class
class QianWenChatLLM(LLM):
    max_length = 10000
    temperature: float = 0.01
    top_p = 0.9

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self):
        return "ChatLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(prompt)
        response, history = model.chat(tokenizer, prompt, history=None)
        torch_gc()
        return response
    
# create the qwen llm

llm = QianWenChatLLM()
print('@@@ qianwen LLM created')
llm("hello")