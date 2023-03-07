
"""
# Dataset information

data = [
        {
          'author': '---', 
          'category': '------', 
          'category_bn': '----', 
          'published_date': '----', 
          'modification_date': '---', 
          'tag': '----', 
          'comment_count': '---',
          'title': '----', 
          'url': '----', 
          'content':"----"`
          }
        ]
"""

import os
import json


MAX_CONTENT = 50

def data_processing(json_path:str="",  output_path:str=""):
  #   with open(file) as f:
  # jdata = json.load(f)
  data = json.load(open(json_path))
  cnt = 0
  cont_list = []
  marge_file_name = "news_dataset.txt"
  with open(os.path.join(output_path, marge_file_name), "w", encoding='utf-8') as mf:
    for i in data:
      content = i["content"]
      cont_list.append(content)
      if len(cont_list)== MAX_CONTENT:
        print("cont : ", cnt)
        filename = f"news{cnt}.txt"
        with open(os.path.join(output_path, filename), "w", encoding='utf-8') as f:
          for ct in cont_list:
            f.write(ct+"\n")
            mf.write(ct+"\n")
        cont_list = []
        cnt+=1


if __name__ =="__main__":
  json_path = "./data/news_paper/data_v2.json"
  output_path = "./data/news_paper_txt"
  os.makedirs(output_path, exist_ok=True)
  data_processing(json_path= json_path, output_path= output_path)

  
