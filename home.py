import streamlit as st
import pandas as pd
import numpy as np
import requests
import json


session_stats = st.session_state

st.set_page_config(
   page_title="个人AI中心",
   page_icon="📝",
   layout="wide",
   initial_sidebar_state="collapsed",
)

options = ["KG-RAG", "WEB"]

cos_url = "http://0.0.0.0:11073/api/v2/cal_cos"

headers = {
    "Content-Type": "application/json"
}
st.header("这个页面用来做web搜索")

def local_req(url, data, timeout=12):
   response = requests.post(url, json=data, headers=headers, timeout=timeout)
   print("resp-------:", response)
   resp = response.json()['resp']
   return resp


# 本来应该写成一个函数，一步直接返回，但是这里还是先分步吧

if True:
   # 输入
   input_str = st.text_input(label="👇在这里输入问题", placeholder="输入想要提问的内容", max_chars=40000)

   # 抽取关键词 
   if st.button("提问！"):
      if input_str is not None and len(input_str) >0:
         with st.expander(label="抽取关键词", expanded=True):
            with st.empty():
               url = "http://0.0.0.0:11073/api/v2/keyword_extract"
               data = {'input_str': input_str}
               print("data:", data)
               try:
                  resp = local_req(url, data)
               except:
                  resp = input_str
               st.markdown(resp)
               key_words = resp.replace("KEYWORDS: ", "")
         # 调用web search接口
         st.write("search baidu.com ...") 
         url = "http://0.0.0.0:11074/api/web/baidu_search"
         data = {"key_words": key_words, "query": input_str + " 最新 2024"}
         try:
            result_info = local_req(url, data, timeout=60)
         except:
            st.write("baidu failed !!!")
            st.stop()

         web_df = pd.DataFrame(result_info[:6], columns=['url','title', 'summary', 'score'])
         st.dataframe(web_df, use_container_width=True)
         st.title("\n look up most relative web ...")
         search_list = []
         web_cnt = 0
         # 尝试阅读最相关的三个网页，并概括网页内容
         for idx, web_info in enumerate(result_info):
            web_score, web_title, web_sum, web_url = web_info
            date_time = web_sum.strip().split(" ")[0]
            if date_time[:4] in ("2023", '2022', '2021', '2022'):
               print("date_time:", date_time)
               continue
            if web_cnt >= 3:
               search_list.append([float(web_score), web_title + "：" + web_sum + "。\n"])
               break
            if web_score > 0.3:
               # try look up web
               try:
                  st.write("lookin :", web_url)
                  st.write("lookin :", web_title)
                  url = "http://0.0.0.0:11074/api/web/browse_web"
                  data = {"url": web_url}
                  web_txt = local_req(url, data, timeout=60)
                  if web_txt == "":
                     continue
                  print("web_look:", web_txt)
                  # 概括内容
                  url = "http://0.0.0.0:11073/api/v2/query_summ"
                  data = {"input_str": input_str, "context_txt": web_txt}
                  web_txt = local_req(url, data, timeout=120)
                  st.write(web_title + "\n" + web_txt + "\n\n")
                  web_cnt += 1
                  # result_info[idx][2] += "。" + result_info
               except:
                  web_txt = ""
                  print("====failed to look web ", web_url)
               print("web_score:", web_score)
               search_list.append([float(web_score), web_title + "：" + web_sum + "。\n" + web_txt])
            else:
               break
            

         context_txt = ""
            
         # 精排
         st.write("------------", len(search_list))
         print("------------", len(search_list))
         sorted_list = sorted(search_list, key=lambda x:x[0], reverse=True)
         sorted_list = sorted_list[:7]
         # 生成kg_context
         for idx, con_txt in enumerate(sorted_list):
            txt = con_txt[1]
            context_txt += " " + str(idx) + "." + txt + "\n"
         
         with st.expander(label="搜索结果", expanded=True):
            search_df = pd.DataFrame(sorted_list, columns=['score', 'ref_txt'])
            st.dataframe(search_df, use_container_width=True)

         # 处理分析
         st.title("AI 正在生成答案 。。。")
         with st.expander(label="搜索结果", expanded=True):
            with st.empty():
               url = "http://0.0.0.0:11073/api/v2/web_process"
               data = {'input_str': input_str, "context_txt": context_txt}
               print("data:", data)
               resp = local_req(url, data, timeout=120)

               st.markdown(resp)

      else:
         st.markdown("请输入问题")
