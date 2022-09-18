import argparse
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("ashiqabdulkhader/GPT2-Malayalam")
model = TFGPT2LMHeadModel.from_pretrained("ashiqabdulkhader/GPT2-Malayalam")

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--text", type=str, default="മലയാളത്തിലെ പ്രധാന ഭാഷയാണ്")
argparser.add_argument("--max_length", type=int, default=100)
argparser.add_argument("--num_beams", type=int, default=5)
argparser.add_argument("--temperature", type=float, default=0.7)
argparser.add_argument("--no_repeat_ngram_size", type=int, default=2)
argparser.add_argument("--num_return_sequences", type=int, default=5)
args = argparser.parse_args()

encoded_text = tokenizer.encode(args.text, return_tensors='tf')
beam_output = model.generate(
    encoded_text,
    max_length=args.max_length,
    num_beams=args.num_beams,
    temperature=args.temperature,
    no_repeat_ngram_size=args.no_repeat_ngram_size,
    num_return_sequences=args.num_return_sequences
)

print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
