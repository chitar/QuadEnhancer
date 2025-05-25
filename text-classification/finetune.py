import argparse
import os

import layers


parser = argparse.ArgumentParser()
parser.add_argument('--saw_method', type=str, default='roll')
parser.add_argument('--saw_k', type=int, default=1)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--model', type=str, default='GPT2-Micro16')
parser.add_argument('--dataset', type=str, default='emotion',help="imdb | yelp_polarity  | ag_news | glue,cola | glue,sst2  | emotion")

args = parser.parse_args()
import transformers
import torch


if args.saw_method != 'linear':
    from layers import SAWLinear, SAWConv1D, saw_method

    layers.saw_method=args.saw_method
    layers.saw_k=args.saw_k

    torch.nn.Linear = SAWLinear
    transformers.pytorch_utils.Conv1D = SAWConv1D


from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2ForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from config import MODEL_FACTORY
from sklearn.metrics import accuracy_score


if ',' in args.dataset:
    dataset = load_dataset(*args.dataset.split(','))
else:
    dataset = load_dataset(args.dataset)


pre_dir = f"./models/{args.model}-{args.saw_method}-{args.saw_k}"
tokenizer = GPT2Tokenizer.from_pretrained(pre_dir)
tokenizer.pad_token = tokenizer.eos_token


def encode(examples):
    # Tokenize the text and create labels (labels = input_ids in this case)
    input_name = 'text' if 'text' in examples.data.keys() else 'sentence'
    encodings = tokenizer(examples[input_name], return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    encodings['labels'] = examples['label']

    return encodings


train_data = dataset['train'].map(encode, batched=True,num_proc=12)
test_name = 'validation' if 'validation' in dataset.keys() else 'test'
test_data = dataset[test_name].map(encode, batched=True,num_proc=6)


n_class = len(set(train_data['labels']))
model = GPT2ForSequenceClassification.from_pretrained(pre_dir,num_labels=n_class)
model.config.pad_token_id = tokenizer.eos_token_id

# initialize saw layers
state_dict = model.state_dict()
for k,v in state_dict.items():
    if 'theta' and 'score' in k:
        torch.nn.init.zeros_(state_dict[k])
model.load_state_dict(state_dict)


out_dir = f"./finetune-models/{args.model}-{args.saw_method}-{args.saw_k}-{args.dataset}"
training_args = TrainingArguments(
    output_dir=out_dir,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.0,
    logging_dir=out_dir,
    logging_steps=1000,
    learning_rate=5e-5,
    # lr_scheduler_type="cosine",
    optim="adamw_torch",
    eval_strategy='epoch',
    save_strategy='epoch',
    # eval_steps=1000,
    # save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    fp16=True,
    disable_tqdm=False,
    save_total_limit=3,
    dataloader_num_workers=2,
    use_cpu=False
)


def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics

)


trainer.train()



# save logs
with open(os.path.join(out_dir, 'train_log.txt'), 'w') as f:
    for line in trainer.state.log_history:
        f.write(f"{line.__str__()}\n")
    f.write(f'Best accuracy: {trainer.state.best_metric}\n')
# save state
trainer.save_state()

model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print(f"Best model saved")


