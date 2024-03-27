from transformers import AutoTokenizer, AutoModel
import chatglm_cpp
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


class RAGStringQueryEngine():
    def __init__(
            self,
            # model_path: str = "./models/chatglm3-6b-32k",
            emb_path: str = "./models/all-MiniLM-L6-v2",
            model_path = "./models/chatglm3-6b-32k-int4/chatglm3-32k-ggml-q4_0.bin",
            faiss_demension: int = 512,
            # PERSIST_DIR: str = "./storage",
            ) -> None:
        
        self.emb_path = emb_path
        self.emb_model = SentenceTransformer(self.emb_path, trust_remote_code=True)
        self.model_path = model_path
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
        self.pipe = chatglm_cpp.Pipeline(self.model_path)
        # self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, load_in_8bit=True)
        self.faiss_demension = faiss_demension
        
    
    def cal_cos(self, text1, text2):
        vt1 = self.emb_model.encode(text1)
        vt2 = self.emb_model.encode(text2)
        return {"resp": 1-cosine(vt1, vt2)}
    
    def raw_query(self, query_str, user="default"):
        resp = self.pipe.generate(query_str)
        print("trans resp:", resp)
        return {"resp": resp}

    def keyword_ext(self, query_str, user="default"):
        max_trip=10
        prompt_template = (
            "下面提供了一个问题。根据提供的问题, 抽取最多{max_keyword}条关键词。"
            "注意，尽力抽取有助于寻找问题答案的关键词. Avoid stopwords.\n"
            "如果文本中不存在关键词，那么什么都不要回复。"
            "注意，只抽取问题中存在的关键词，不要返回或者联想问题中不存在的关键词！也不要尝试回答问题！"
            "绝对不要返回超出问题字面的信息，专注于问题文本本身！"
            "---------------------\n"
            "问题：{question}\n"
            "---------------------\n"
            "关键词使用这样的格式返回: 'KEYWORDS: <keywords>'\n"
        )
        prompt_template_str = "".join(prompt_template)
        text2chatglm = prompt_template_str.format_map({
            'max_keyword': max_trip,
            'question': query_str,
        })
        resp = self.pipe.generate(text2chatglm)
        print("trans resp:", resp)
        return {"resp": resp} 

    def web_process(self, query_str, context_txt, user="default"):
        # max_trip=10
        prompt_template = (
            "下面提供了一个问题。根据提供的问题, 和已知的上下文内容，简洁和专业的来回答用户的问题。"
            "---------------------\n"
            "问题：{question}\n"
            "---------------------\n"
            "已知上下文：{context_txt}\n"
            "---------------------\n"
        )
        
        prompt_template_2 = (
            "基于搜索结果，结合自身背景知识，回答问题『{question}』，在开头给出核心观点或答案。"
            "如果涉及不同维度的对比，优先使用表格进行展示，列展示对比的维度。\n"
            "回答遵循要求：\n"
            "  1. 尽量使用换行、列表样式等格式来组织答案，回答时尽量避免信息冗余，控制在500字以内。\n"
            "  2. 如果回答中内容较多时，尽量采用总分总格式，即开头给出观点、中间分维度进行阐述、总结概括收尾。\n"
            "  3. 如果回答是分维度阐述，维度需要全面、精准。\n"
            "  4. 如果回答中涉及不同维度的对比，优先尽量采用表格进行展示，列展示对比的维度。\n"
            "  5. 注意采纳较新的信息，如果新旧信息冲突，优先采用更新的消息。 \n"
            "搜索结果：\n{context_txt}"
        )
        
        prompt_template_str = "".join(prompt_template)
        prompt_template_str_2 = "".join(prompt_template_2)
        # text2chatglm = prompt_template_str.format_map({
        #     'context_txt': context_txt,
        #     'question': query_str,
        # })
        text2chatglm = prompt_template_str_2.format_map({
            'context_txt': context_txt,
            'question': query_str,
        })

        print("------------------")
        print(text2chatglm)
        print("------------------")
        resp = self.pipe.generate(text2chatglm)
        print("trans resp:", resp)
        return {"resp": resp} 

    def query_summ(self, query_str, content_txt, user="default"):
        prompt_template = (
            "下面提供了一个问题，和一段文章。根据提供的问题, 概括文章相关内容，不要超过400字。"
            "注意，尽力抽取、概括文章中，有助于寻找问题答案的信息. Avoid stopwords.\n"
            "尽量避免重复的信息和描述。"
            "注意，只抽取问题中存在的关键词，不要返回或者联想问题中不存在的关键词！也不要尝试回答问题！"
            "绝对不要返回超出问题字面的信息，专注于提供的文本本身！"
            "---------------------\n"
            "问题：{question}\n"
            "---------------------\n"
            "待概括的文章：\n{context}"
            "---------------------\n"
            "概括结果：\n"
        )
        prompt_template_str = "".join(prompt_template)
        text2chatglm = prompt_template_str.format_map({
            'content': content_txt,
            'question': query_str,
        })
        resp = self.pipe.generate(text2chatglm)
        print("trans resp:", resp)
        return {"resp": resp} 

if __name__ == "__main__":
    test = RAGStringQueryEngine()
    # test.query_raw("项目相关人员都有谁，列出联系方式")