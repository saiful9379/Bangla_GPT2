"""
conda activate sufia
"""

import os
import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME

from pathlib import Path


save_path = "tokenized_data"
data_path = "./data/bn_corpus/"

paths = [str(x) for x in Path(data_path).glob("**/*.txt")][:10000]
print(len(paths))
tokenizer = GPT2Tokenizer.from_pretrained(save_path)
tokenizer.add_special_tokens({
  "eos_token": "</s>",
  "bos_token": "<s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "mask_token": "<mask>"
})# creating the configurations from which the model can be made
config = GPT2Config(
  vocab_size=tokenizer.vocab_size,
  bos_token_id=tokenizer.bos_token_id,
  eos_token_id=tokenizer.eos_token_id
)# creating the model
model = TFGPT2LMHeadModel(config)

single_string = 'বাংলা ভাষা আন্দোলন তদানীন্তন পূর্ব পাকিস্তানে (বর্তমান বাংলাদেশ) সংঘটিত একটি সাংস্কৃতিক ও রাজনৈতিক আন্দোলন। মৌলিক অধিকার রক্ষাকল্পে বাংলা ভাষাকে ঘিরে সৃষ্ট এ আন্দোলনের মাধ্যমে তৎকালীন পাকিস্তানের অন্যতম রাষ্ট্রভাষা হিসেবে প্রতিষ্ঠার লক্ষ্যে গণদাবীর যথাযথ প্রতিফলন ঘটে।'
# for filename in paths[:3]:
#   with open(filename, "r", encoding='utf-8') as f:
#    x = f.read()
  #  print(x)
  #  if len(x)==0:
  #   print(x)  
single_string += x + tokenizer.eos_token
string_tokenized = tokenizer.encode(single_string)

print(string_tokenized)
token_list = [tokenizer.decode(i) for i in string_tokenized]
print(token_list)
