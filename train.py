import os
import pandas as pd

import datasets
from datasets import load_dataset, Dataset, DatasetDict

import torch, numpy
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_scheduler

from accelerate import Accelerator

import nltk
import numpy as np
from nltk import sent_tokenize
from rouge_chinese import Rouge
import jieba

from tqdm.auto import tqdm

from utils import RougeScoreChinese, postprocess_text

os.environ['TRANSFORMERS_OFFLINE'] = '1'
data_path = './data/LCSTS_csv'
source_model = "./hub/mt5-small-finetuned-on-mT5-lcsts"

tokenizer = AutoTokenizer.from_pretrained("./hub/mt5-small-finetuned-on-lcsts")
model = AutoModelForSeq2SeqLM.from_pretrained("./hub/mt5-small-finetuned-on-lcsts")

def preprocess_function(examples, tokenizer=tokenizer):
    max_input_length = 512
    max_target_length = 30

    model_inputs = tokenizer(
        examples["text"],
        max_length=max_input_length,
        truncation=True,
        padding=True,
    )
    labels = tokenizer(
        examples["target"], 
        max_length=max_target_length, 
        truncation=True,
        padding=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = DatasetDict.from_csv({'train': '{}/eval.csv'.format(data_path), 'validation': '{}/test.csv'.format(data_path)})
tokenized_datasets = dataset.map(preprocess_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)  # 删除原本的列名

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


tokenized_datasets.set_format("torch")

batch_size = 16
train_dataloader = DataLoader(tokenized_datasets["validation"], shuffle=True, collate_fn=data_collator, batch_size=batch_size)
eval_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=batch_size)


optimizer = AdamW(model.parameters(), lr=2e-5)


accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps,)

    
rouge_score = RougeScoreChinese()


output_dir = "results-mt5-finetuned-squad-accelerate"
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=1000,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            # If we did not pad to max length, we need to pad the labels too
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )
            # print('predictions={}, \nreferences={}'.format(decoded_preds, decoded_labels))
            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Compute metrics
    result = rouge_score.compute()
    # print(result)
    # Extract the median ROUGE scores
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    # result = {k: round(v, 4) for k, v in result.items()}

    scores = dict()
    for name in result.keys():
        res = {k: round(v, 4) for k, v in result[name].items()}
        scores[name] = res

    print(f"Epoch {epoch}:", scores)

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
