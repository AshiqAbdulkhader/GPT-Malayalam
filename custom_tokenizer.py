# create custom tokenizer for gpt2 for malayalam
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFD, Sequence
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from pathlib import Path
import os

# Initialize a tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = Sequence([NFD()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

# Customize training
trainer = BpeTrainer(vocab_size=50000, show_progress=True, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>"
])

# data
paths = [str(x) for x in Path("./ml_corpus/").glob("**/*.txt")]

# Train the tokenizer
tokenizer.train(files=paths, trainer=trainer)

save_path = 'tokenized_data'
if not os.path.exists(save_path):
    os.makedirs(save_path)
tokenizer.model.save(save_path)
