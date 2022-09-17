# test GPT2 malayalam model
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


output_dir = "gpt2_malayalam"
# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
# load model
model = TFGPT2LMHeadModel.from_pretrained(output_dir)

text = "മലയാളത്തിലെ പ്രധാന ഭാഷയാണ്"
input_ids = tokenizer.encode(text, return_tensors='tf')

beam_output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    temperature=0.7,
    no_repeat_ngram_size=2,
    num_return_sequences=5
)

print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
