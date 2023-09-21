import chromadb
from transformers import AutoTokenizer, AutoModel
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from chromadb.config import Settings
model = Model.from_pretrained('ZhipuAI/chatglm2-6b', device_map='auto', revision='v1.0.9')

class ModelScopWrapper():
        def __init__(self,db_path,dbname,model,textvetor_model_id="damo/nlp_corom_sentence-embedding_chinese-base"):
            self.pipeline_textvetor = pipeline(Tasks.sentence_embedding,model=textvetor_model_id)

            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                anonymized_telemetry=False,
                persist_directory=db_path # Optional, defaults to .chromadb/ in the current directory
            ))
            self.db = self.client.get_or_create_collection(dbname)
            self.splitter = ChineseTextSplitter(pdf=False, sentence_size=1024)
            pass

        def put_data(self,text_list):
            input_text = [] 
            ids = []
            total = len(text_list)
            index = self.db.count()
            if not index:
                index = 0
            for ttt_i in range(total):
                input_text.append(text_list[ttt_i])
                index+= 1
                ids.append(str(index))
                if len(input_text) > 1000:
                    print("%d / %d", ttt_i, total)
                    self.__insert_db(input_text,ids)
                    input_text = []
                    ids = []
            if len(ids) > 0:
                print("%d / %d", ttt_i, total)
                self.__insert_db(input_text,ids)
            self.client.persist()

        def __insert_db(self,input_text,ids):
            inputs = {
                    "source_sentence": input_text
                }
            result = self.pipeline_textvetor(input=inputs)
            self.db.add(embeddings=result['text_embedding'].tolist(), documents=input_text,ids=ids) 


        def __vector_query(self,text, top=5):
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
        def load_file_to_db(self, file):
            with open(file, encoding= 'utf-8') as f:
                book = f.read()

            inputs = self.splitter.split_text(book)
            self.put_data(inputs)


        def query(self, text, top=5):
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

w.load_file_to_db("story.txt")
w.query("谢俞是一个人物,介绍一下关于他的事情")