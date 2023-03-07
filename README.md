# Bangla_GPT2

OpenAI GPT-2 model was proposed in Language Models are Unsupervised Multitask Learners. This OpenAI GPT2 model was train using Bangla News paper dataset . Here  we used prothom algo 250mb data for GPT2 model training purpose.

# Requirements,

```
tensorflow-gpu==2.6.1
transformers==4.22.1
tokenizers==0.12.1
torch==1.11.0+cu113  
```
Model Description

OpenAI GPT-2 model was proposed in Language Models are Unsupervised Multitask Learners paper .Original GPT2 model was a causal (unidirectional) transformer pretrained using language modeling on a very large corpus of ~40 GB of text data. This model has same configuration but has been pretrained on bengali corpus of mC4(multilingual C4) dataset. The code for training the model has all been open-sourced here.
Training Details

Overall Result:

Eval loss : 1.45, Eval Perplexity : 3.141

Data: mC4-bn

Train Steps: 250k steps

link 🤗 flax-community/gpt2-bengali

Demo : https://huggingface.co/spaces/flax-community/Gpt2-bengali
Usage

For using the model there are multiple options available. For example using the pipeline directly we can try to generate sentences.
