from fastapi import FastAPI, Request
import uvicorn, json
from pyppeteer import launch
from pyquery import PyQuery as pq
import requests
import time
import re


width, height = 1366, 768
cos_url = "http://0.0.0.0:11073/api/v2/cal_cos"
headers = {
    "Content-Type": "application/json"
}
app = FastAPI()


def local_req(url, data):
    response = requests.post(url, json=data, headers=headers)
    resp = response.json()['resp']
    return resp

CLEANR = re.compile('<.*?>')
CLEANR_2 = re.compile('{.*?}')
CLEANR_3 = re.compile(".site-.* ")
def remove_html_tags(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    cleantext = re.sub(CLEANR_2, '', cleantext)
    cleantext = re.sub(CLEANR_3, '', cleantext)
    return cleantext

@app.post("/api/web/baidu_search")
async def key_search(request: Request):
    # global bw
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    wd = json_post_list.get('key_words')
    query = json_post_list.get('query')
    browser = await launch(
        headless=True,
        userDataDir='./userdata',
        args=[f'--window-size={width},{height}','--download-path=./'])
        # args=[f'--window-size={width},{height}','--proxy-server=http://192.168.31.101:10792','download-path=./'])
    page = await browser.newPage()
    await page.setViewport({'width': width, 'height': height})
    await page.goto('https://www.baidu.com/s?wd=' + wd)
    await page.waitForSelector('body');
    
    doc = pq(await page.content())
    # print(len(doc('h3')))
    mid_result = []
    all_desc = {}
    for d in doc('div.c-container').items():
        # print("----------------")
        for ad in d('a').items():
            # print(ad.attr('href'))
            d_url = ad.attr('href')
            break
        
        d_title = d('h3').text()
        if d_title in all_desc:
            continue
        else:
            all_desc[d_title] = True
        d_sum = d('span').text()
        # print(d_url, d_title, d_sum)
        d_desc = d_title + "：" + d_sum
        data = {"text_1": d_desc, "text_2": query}
        t_score = local_req(cos_url, data)
        # if t_score < 0.3:
        #     continue
        # print("--------===============")
        # print(d_url, d_title, d_sum, t_score)
        mid_result.append([t_score, d_title, d_sum, d_url])
        
    # 排序
    if len(mid_result) == 0:
        return ""
    sorted_list = sorted(mid_result, key=lambda x:x[0], reverse=True)
    # browser.close()
    return {"resp": sorted_list}

@app.post("/api/web/browse_web")
async def key_search(request: Request):
    global bw
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    url = json_post_list.get('url')

    print("url:", url)
    if url[:4] != "http":
        return {"resp": ""}
    # query = json_post_list.get('query')
    
    # result = []
    browser = await launch(
        headless=True,
        userDataDir='./userdata',
        # args=[f'--window-size={width},{height}','--download-path=./'])
        args=[f'--window-size={width},{height}', '--download-path=./'])
    page = await browser.newPage()
    await page.setViewport({'width': width, 'height': height})
    await page.goto(url)
    time.sleep(6)
    await page.waitForSelector('body');
    time.sleep(4)
    most_doc = pq(await page.content()) 
    most_text = most_doc('p').text()[:4000]
    most_text = remove_html_tags(most_text)
    # title = web_info[1]
    # browser.close()
    if len(most_text) > 200:
        return {"resp": most_text}
    else:
        return {"resp": ""}

    
if __name__ == "__main__":
    # bw = bwser()
    uvicorn.run(app, host='0.0.0.0', port=11074, workers=1)