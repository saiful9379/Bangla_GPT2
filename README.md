# Bangla_GPT2

OpenAI GPT-2 model was proposed in Language Models are Unsupervised Multitask Learners. This OpenAI GPT2 model was train using Bangla News paper dataset . Here  we used prothom algo 250mb data for GPT2 model training purpose and also We use vocab size 50k for this model. 
Original GPT2 model was a causal (unidirectional) transformer pretrained using language modeling on a very large corpus of ~40 GB of text data. This model has same configuration but has been pretrained on bengali corpus of mC4(multilingual C4) dataset. 

🤗 ![Demo in huggingface](https://huggingface.co/saiful9379/Bangla_GPT2)
# Requirements,
```
tensorflow-gpu==2.6.1
transformers==4.22.1
tokenizers==0.12.1
torch==1.11.0+cu113  
```
# Download Dataset,
wiki data download using below this script. this script save the data chunk by chunk.
Run,
```
python wikipedia_download.py --lang bn
```
# Model configuration,
Here the basic model configuration,
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
