
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


def load_models(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = TFGPT2LMHeadModel.from_pretrained(model_path)
    return model, tokenizer


def prediction(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors='tf')
    # print(tokenizer.decode(input_ids[0], skip_special_tokens=True))

    predicted_text = model.generate(
        input_ids,
        max_length=175,
        num_beams=10,
        temperature=0.7,
        no_repeat_ngram_size=2,
        num_return_sequences=5
    )
    return tokenizer.decode(predicted_text[0], skip_special_tokens=True)

if __name__ =="__main__":
    model_path = "./gpt2_bangla"
    model, tokenizer = load_models(model_path)
    text = "গাজীপুরের কালিয়াকৈর উপজেলার তেলিরচালা"
    predicted_text = prediction(text, model, tokenizer)
    print(predicted_text)