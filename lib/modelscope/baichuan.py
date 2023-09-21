import torch
from modelscope import snapshot_download, Model
model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat-4bits", revision='v1.0.0') #specify and download the llm model
model = Model.from_pretrained(model_dir, device_map="balanced", trust_remote_code=True, torch_dtype=torch.float16)

class ModelScopWrapper():
        def __init__(self,file_pah,model,
                 textvetor_model_id="damo/nlp_corom_sentence-embedding_chinese-base"
                ):
            self.pipeline_textvetor = pipeline(Tasks.sentence_embedding,model=textvetor_model_id)

            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                anonymized_telemetry=False,
                persist_directory=file_pah # Optional, defaults to .chromadb/ in the current directory
            ))
            self.db = self.client.get_or_create_collection('modelscope')
            self.splitter = ChineseTextSplitter(pdf=False, sentence_size=1024) # use customized chinese text splitter to split the contents of load fie
    
        
        #put knowledge data to db
        def put_data(self,text_list):
            input_text = [] 
            ids = []
            index = self.db.count()
            total = len(text_list)
            if not index:
                index = 0
            for ttt_i in range(total):
                input_text.append(text_list[ttt_i])
                index+=1
                ids.append(str(index))
                if len(input_text) > 1000:
                    print("%d / %d", ttt_i, total)
                    self.__insert_db(input_text,ids)
                    input_text = [] 
                    ids = []
            if len(ids) > 0:
                print("%d / %d", ttt_i, total)
                self.__insert_db(input_text,ids)
            print(self.db.count())
                
        
        def __insert_db(self,input_text,ids):
            inputs = {
                    "source_sentence": input_text
                }
            result = self.pipeline_textvetor(input=inputs)
            self.db.add(embeddings=result['text_embedding'].tolist(), documents=input_text,ids=ids) 
            
        # query embedding results from vecotr db
        def __vector_query(self,text,top):
            db= self.db
            input_text = [ text ]
            inputs = {
                    "source_sentence": input_text
                }
            result = self.pipeline_textvetor(input=inputs)

            result = db.query(
                query_embeddings=result['text_embedding'].tolist()
                ,n_results=top
            )
            return result
        
        # put knowledge data from file to vector db
        def load_file_to_db(self, file):
            with open(file, encoding= 'utf-8') as f:
                book = f.read()

            inputs = self.splitter.split_text(book)
            self.put_data(inputs)
        
        
        # query results from knowldge base via llm
        def query(self, text, top):
            results = self.__vector_query(text, top)
            contentext ='。\n'.join(results['documents'][0])+"."
            qa = """请根据上下文来回答问题，如果根据上下文的内容无法回答问题，请回答"我不知道"。不需要编造信息。
                    上下文：
                    ```
                    """+contentext+"""
                    ```

                    问题：```"""+text+"""？```

                    """
            messages = []
            messages.append({"role": "user", "content":qa})
            response = model(messages)
            return response
w = ModelScopWrapper("/mnt/workspace/chroma", model)
w.query("my question", 5) # ask question from top 5 results from vector db