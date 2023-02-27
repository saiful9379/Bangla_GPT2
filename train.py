import os
import argparse
import tensorflow as tf
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFD, NFKC,Sequence
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import GPT2Tokenizer, GPT2Config, TFGPT2LMHeadModel
from transformers import WEIGHTS_NAME, CONFIG_NAME

BLOCK_SIZE = 100
BATCH_SIZE = 24
BUFFER_SIZE = 1000

def tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list:
    return [token for token in text.split() if token_min_len <= len(token) <= token_max_len]

def train_bpe_tokenizer(opt)->None:
    data_list = [str(x) for x in Path(opt.data).glob("*.txt")]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=50000, 
        show_progress=True,
        inital_alphabet=ByteLevel.alphabet(),
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    tokenizer.train(files=data_list, trainer=trainer)
    tokenizer.model.save(opt.tokenizer)

def load_tokenizer(opt):
    tokenizer = GPT2Tokenizer.from_pretrained(opt.tokenizer,  unk_token="[UNK]")
    return tokenizer

def load_data(opt, tokenizer):
    paths = [str(x) for x in Path(opt.data).glob("*.txt")][:3000]
    # paths = [for i in os.listdir(opt.data)]
    single_string = ''
    for filename in paths:
        with open(filename, "r", encoding='utf-8') as f:
            x = f.read()
        single_string += x + tokenizer.eos_token

    string_tokenized = tokenizer.encode(single_string)
    print("Done tokenizing")
    return string_tokenized

def train_gpt(opt):
    tokenizer = load_tokenizer(opt)
    tokenizer.add_special_tokens(
        {"eos_token": "</s>", "bos_token": "<s>", "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>"}
      )
    config = GPT2Config(
      vocab_size=tokenizer.vocab_size, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
      )
    # ================= data information =================================
    examples = []
    string_tokenized = load_data(opt, tokenizer)
    for i in range(0, len(string_tokenized) - BLOCK_SIZE + 1, BLOCK_SIZE):
        examples.append(string_tokenized[i:i + BLOCK_SIZE])

    inputs, labels = [], []
    for ex in examples:
        inputs.append(ex[:-1])
        labels.append(ex[1:])

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(opt.batch, drop_remainder=True)
    print("Done creating dataset")
     # ================= data information =================================

    model = TFGPT2LMHeadModel(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])
    history = model.fit(dataset, epochs=opt.epoch, verbose=1)

    #========================== Save model =================================
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(opt.save, WEIGHTS_NAME)
    output_config_file = os.path.join(opt.save, CONFIG_NAME)
    model.save_pretrained(opt.save)
    model_to_save.config.to_json_file(output_config_file)
    # save tokenizer
    tokenizer.save_pretrained(opt.save)


if __name__ =="__main__":

  # data_path = "dataset/news_dataset.txt"
  # tokenizer_save_path = "tokenizer_voc"
  # save_model = "bangla_gpt2"
  # os.makedirs(tokenizer_save_path, exist_ok= True)
  # os.makedirs(save_model, exist_ok= True)
  """
  python train.py --data dataset/news_paper_txt --tokenizer tokenizer_voc --save bangla_gpt2
  """

  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int, default=20, help='save model path')
  parser.add_argument('--batch', type=int, default=16, help='save model path')
  parser.add_argument('--tokenizer', type=str, default='tokenizer_voc', help='tokenizer_voc path(s)')
  parser.add_argument('--data', type=str, default='dataset/news_paper_txt', help='data')  # file/folder, 0 for webcam
  parser.add_argument('--vocab_size', type=int, default=50000, help='voc size')
  parser.add_argument('--save', type=str, default="bangla_gpt2", help='save model path')
  opt = parser.parse_args()

  print(f"Args : {opt}")

  print("Tokenizer Training ................ : ", end="", flush=True)
  train_bpe_tokenizer(opt)
  print("Done")
  print(" Training Bangla GPT2 Model................ : ")
  train_gpt(opt)