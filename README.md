# Bangla_GPT2
Bangla GPT2 model was trained using the Bangla Newspaper dataset. Here we used prothom alo 250mb data for GPT2 model training and also vocab size 50k. 

🤗 ![Demo in huggingface](https://huggingface.co/saiful9379/Bangla_GPT2)
# Requirements,
```
tensorflow-gpu==2.6.1
transformers==4.22.1
tokenizers==0.12.1
torch==1.11.0+cu113  
```
# Download Dataset,
Download the wiki data Run,
```
python wikipedia_download.py --lang bn
```
# Model configuration,
Here the basic configuration of Bangla GPT2 model,
```
vocab_size = 50000
block_size = 200
learning_rate=3e-5
num_epoch = 100
batch_size = 12
buffer_size = 1000
```
# Train
For training GPT2 model,

```
python train.py --data dataset/news_paper_txt --tokenizer tokenizer_voc --save bangla_gpt2
or
example/train-gpt-2-Bangla-language-model.ipynb
```
# Inference

```
inference.ipynb
```


Overall Result:
Perplexity : 6.7
