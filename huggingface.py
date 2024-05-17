import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

TRANSFORMERS_OFFLINE=1

# pipeline

# text = '接诊医生为患者测量空腹血糖，血糖条无法自动吸血，导致无法测量，更换新的血糖条后，顺利为患者测量了空腹血糖。'
# summarizer = pipeline("summarization", model="./hub/mt5-small-finetuned-on-mT5-lcsts")

# print(summarizer(text, max_length=30, min_length=10, do_sample=False))

# Normal
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

# source_model = "./hub/mt5-small-finetuned-on-mT5-lcsts"

# # tokenizer = AutoTokenizer.from_pretrained("./hub/bert-base-chinese", cache_dir = './hub/bert-base-chinese')
# # model = AutoModelForMaskedLM.from_pretrained("./hub/bert-base-chinese", cache_dir = '.hub/bert-base-chinese')
# tokenizer = AutoTokenizer.from_pretrained("./hub/mt5-small-finetuned-on-mT5-lcsts", cache_dir = './hub/mt5-small-finetuned-on-mT5-lcsts')
# model = AutoModelForSeq2SeqLM.from_pretrained("./hub/mt5-small-finetuned-on-mT5-lcsts", cache_dir = '.hub/mt5-small-finetuned-on-mT5-lcsts')



import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate


# "IDEA-CCNL/Wenzhong-GPT2-110M"
#  "IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese"
model_name = "./hub/mt5-small-finetuned-on-mT5-lcsts"
data_path = './data/LCSTS_csv/'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token


raw_datasets = load_dataset('csv', data_files={'train': '{}test.csv'.format(data_path), 'test': '{}eval.csv'.format(data_path)})

max_input_length = 128
max_target_length = 20
batch_size = 8
prefix = "请问下面文本属于 招聘信息、 经验贴、 求助贴 三者中的哪一类？\n"
suffix = "\n选项：招聘信息, 经验贴, 求助贴\n答案：\n"

def preprocess_function(examples):

    inputs = [prefix + doc + suffix for doc, tg in zip(examples["text"], examples["target"])]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding=True, truncation=True)
    model_inputs["labels"] = tokenizer(examples["target"], max_length=max_input_length, padding=True, truncation=True)['input_ids']
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)




args = TrainingArguments(
    f"mt5-finetuned-lcsts",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 10,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    save_total_limit=1,
    num_train_epochs=20,
    # predict_with_generate=True,
    load_best_model_at_end=True,
    fp16=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer,  mlm=False)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)

trainer.train()
print("test")
print(trainer.evaluate())
trainer.save_model("mt5_lcsts")

# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='mygpt')
# set_seed(42)
# s = generator("请问下面文本属于 招聘信息、 经验贴、 求助贴 三者中的哪一类？\nviv0社招。 	#春招# 有匹配岗位 有意向大佬欢迎＋微g1r4ffe内推 ...viv0社招开启，岗位多多hc多多。博士应聘专家岗位有1年以上工作经验即可 #社招#\n选项：招聘信息, 经验贴, 求助贴\n答案：\n", max_length=256, num_return_sequences=1)
# print(s)

