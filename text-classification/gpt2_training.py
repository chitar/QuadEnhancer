import argparse
import os

import math

import layers


parser = argparse.ArgumentParser()
parser.add_argument('--saw_method', type=str, default='1')
parser.add_argument('--saw_k', type=int, default=0)
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--model', type=str, default='GPT2-Micro16')
parser.add_argument('--extend_data', type=int, default=2)

args = parser.parse_args()
import transformers
import torch


if args.saw_method != 'linear':
    from layers import SAWLinear, SAWConv1D, saw_method

    layers.saw_method=args.saw_method
    layers.saw_k=args.saw_k

    torch.nn.Linear = SAWLinear
    transformers.pytorch_utils.Conv1D = SAWConv1D


from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,concatenate_datasets
from config import MODEL_FACTORY



dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
wiki103_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
# dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def encode(examples):
    # Tokenize the text and create labels (labels = input_ids in this case)
    encodings = tokenizer(examples['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    encodings['labels'] = encodings['input_ids'].clone()  # Make labels equal to input_ids
    return encodings


total_train_dataset =  concatenate_datasets([dataset['train'], wiki103_dataset['train'].select(range(len(dataset['train'])*(args.extend_data-1))) ])
train_data = total_train_dataset.map(encode, batched=True,num_proc=12)
test_data = dataset['validation'].map(encode, batched=True,num_proc=6)


model_cfg = MODEL_FACTORY[args.model]
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=256,
    n_ctx=256,
    n_embd=model_cfg['n_embd'],
    n_layer=model_cfg['n_layer'],
    n_head=model_cfg['n_head']
)

model = GPT2LMHeadModel(config)

# initialize saw layers
state_dict = model.state_dict()
for k,v in state_dict.items():
    if 'theta' and 'score' in k:
        torch.nn.init.zeros_(state_dict[k])
model.load_state_dict(state_dict)



out_dir = f"./models/{args.model}-{args.saw_method}-{args.saw_k}"
training_args = TrainingArguments(
    output_dir=out_dir,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.0,
    logging_dir=out_dir,
    logging_steps=1000,
    learning_rate=1e-4,
    # lr_scheduler_type="cosine",
    optim="adamw_torch",
    eval_strategy='epoch',
    save_strategy='epoch',
    # eval_steps=500,
    # save_steps=500,
    load_best_model_at_end=True,
    # metric_for_best_model='perplexity',
    # greater_is_better=False,
    fp16=True,
    disable_tqdm=False,
    save_total_limit=3,
    dataloader_num_workers=2

)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    # compute_metrics=compute_metrics

)


trainer.train()



# save logs
with open(os.path.join(out_dir, 'train_log.txt'), 'w') as f:
    for line in trainer.state.log_history:
        f.write(f"{line.__str__()}\n")
        f.write(f'Best eval loss: {trainer.state.best_metric}\n')
# save state
trainer.save_state()

model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print(f"Best model saved")


