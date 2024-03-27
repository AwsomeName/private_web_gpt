nohup python apis/all_apis.py 2>&1 >/dev/null &
nohup python apis/ppeter_search_api.py 2>&1 >/dev/null &

streamlit run home.py --server.fileWatcherType none