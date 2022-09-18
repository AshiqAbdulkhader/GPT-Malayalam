from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("ashiqabdulkhader/GPT2-Malayalam")

model = TFGPT2LMHeadModel.from_pretrained(
    "ashiqabdulkhader/GPT2-Malayalam")

text = "മലയാളത്തിലെ പ്രധാന ഭാഷയാണ്"

encoded_text = tokenizer.encode(text, return_tensors='tf')

beam_output = model.generate(
    encoded_text,
    max_length=100,
    num_beams=5,
    temperature=0.7,
    no_repeat_ngram_size=2,
    num_return_sequences=5
)

print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
