
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from langchain import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from torch.mps import empty_cache
import torch
import sys
class GLM(LLM):
    max_token: int = 2048
    temperature: float = 0.8
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 1024

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "GLM"

    def load_model(self, llm_device="gpu",model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True).half().cuda()

    def _call(self,prompt:str,history:List[str] = [],stop: Optional[List[str]] = None):
        response, _ = self.model.chat(
                    self.tokenizer,prompt,
                    history=history[-self.history_len:] if self.history_len > 0 else [],
                    max_length=self.max_token,temperature=self.temperature,
                    top_p=self.top_p)
        return response



modelpath = "/root/model/chatglm-6b"
sys.path.append(modelpath)
llm = GLM()
llm.load_model(model_name_or_path = modelpath)

with open("pirlo.txt") as f:
    report_2023 = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(report_2023)
docs = [Document(page_content=t) for t in texts]
prompt_template = """对下面的文字做精简的摘要:

    {text}

    """

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)
summ = chain({"input_documents": docs}, return_only_outputs=True)
print(summ['output_text'])
