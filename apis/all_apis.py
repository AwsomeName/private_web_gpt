from fastapi import FastAPI, Request
import uvicorn, json
# from rag_faiss_llama_index_quant import RAGStringQueryEngine
from rag_faiss_llama_index import RAGStringQueryEngine
app = FastAPI()


@app.post("/api/v2/cal_cos")
async def cal_cos(request: Request):
    global rag
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    text_1 = json_post_list.get('text_1')
    text_2 = json_post_list.get('text_2')
    output_float = rag.cal_cos(text_1, text_2)

    return output_float

@app.post("/api/v2/raw_query")
async def raw_query(request: Request):
    global rag
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    input_str = json_post_list.get('input_str')
    output_str = rag.raw_query(input_str)

    return output_str    


@app.post("/api/v2/keyword_extract")
async def keyword_ext(request: Request):
    print("load extract!!!")
    global rag
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    input_str = json_post_list.get('input_str')
    print("key_query:", input_str)
    output_str = rag.keyword_ext(input_str)

    return output_str    

    
@app.post("/api/v2/web_process")
async def web_process(request: Request):
    global rag
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    input_str = json_post_list.get('input_str')
    context_txt = json_post_list.get('context_txt')
    output_str = rag.web_process(input_str, context_txt)

    return output_str

@app.post("/api/v2/query_summ")
async def query_summ(request: Request):
    global rag
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    input_str = json_post_list.get('input_str')
    context_txt = json_post_list.get('context_txt')
    output_str = rag.query_summ(input_str, context_txt)

    return output_str
    
    
if __name__ == '__main__':
    rag = RAGStringQueryEngine()
    uvicorn.run(app, host='0.0.0.0', port=11073, workers=1)