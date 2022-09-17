# Train custom GPT2 model for malayalam
import wandb
import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME
from pathlib import Path
import os
# wandb login
wandb.init(project="gpt-2 malayalam", entity="ashiq")

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('tokenized_data', unk_token="[UNK]")
tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})

# load model
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
model = TFGPT2LMHeadModel(config)

# load dataset
paths = [str(x) for x in Path("./ml_corpus/").glob("**/*.txt")]
single_string = ''
for filename in paths:
    with open(filename, "r", encoding='utf-8') as f:
        x = f.read()
    single_string += x + tokenizer.eos_token

# tokenize dataset
string_tokenized = tokenizer.encode(single_string)
print("Done tokenizing")

# create dataset
examples = []
block_size = 100
BATCH_SIZE = 12
BUFFER_SIZE = 1000
for i in range(0, len(string_tokenized) - block_size + 1, block_size):
    examples.append(string_tokenized[i:i + block_size])
inputs, labels = [], []

for ex in examples:
    inputs.append(ex[:-1])
    labels.append(ex[1:])

dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print("Done creating dataset")
# create model
optimizer = tf.keras.optimizers.Adam(
    learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=[
              loss, *[None] * model.config.n_layer], metrics=[metric])

# train model
num_epoch = 10
history = model.fit(dataset, epochs=num_epoch, verbose=1)
wandb.log({"loss": history.history['loss'][-1]})

# save model
save_path = 'gpt2_malayalam'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join(save_path, WEIGHTS_NAME)
output_config_file = os.path.join(save_path, CONFIG_NAME)

model.save_pretrained(save_path)
model_to_save.config.to_json_file(output_config_file)

# save tokenizer
tokenizer.save_pretrained(save_path)
