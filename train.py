import os, argparse
import pandas as pd
import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_scheduler
from datasets import DatasetDict

from accelerate import Accelerator
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

from utils import RougeScoreChinese, postprocess_text

os.environ['TRANSFORMERS_OFFLINE'] = '1'

# data_path = './data/LCSTS_csv'
# source_model = "./hub/mt5-small-finetuned-on-mT5-lcsts"

# tokenizer = AutoTokenizer.from_pretrained("./hub/mt5-small-finetuned-on-lcsts")
# model = AutoModelForSeq2SeqLM.from_pretrained("./hub/mt5-small-finetuned-on-lcsts")

# def preprocess_function(examples, tokenizer=tokenizer):
#     max_input_length = 512
#     max_target_length = 30

#     model_inputs = tokenizer(
#         examples["text"],
#         max_length=max_input_length,
#         truncation=True,
#         padding=True,
#     )
#     labels = tokenizer(
#         examples["target"], 
#         max_length=max_target_length, 
#         truncation=True,
#         padding=True
#     )
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# dataset = DatasetDict.from_csv({'train': '{}/eval.csv'.format(data_path), 'validation': '{}/test.csv'.format(data_path)})
# tokenized_datasets = dataset.map(preprocess_function, batched=True)

# tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)  # 删除原本的列名

# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# tokenized_datasets.set_format("torch")

# batch_size = 16
# train_dataloader = DataLoader(tokenized_datasets["validation"], shuffle=True, collate_fn=data_collator, batch_size=batch_size)
# eval_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=batch_size)


# optimizer = AdamW(model.parameters(), lr=2e-5)


# accelerator = Accelerator()
# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)


# num_train_epochs = 10
# num_update_steps_per_epoch = len(train_dataloader)
# num_training_steps = num_train_epochs * num_update_steps_per_epoch

# lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps,)

    
# rouge_score = RougeScoreChinese()


# output_dir = "results-mt5-finetuned-squad-accelerate"
# progress_bar = tqdm(range(num_training_steps))

# for epoch in range(num_train_epochs):
#     # Training
#     model.train()
#     for step, batch in enumerate(train_dataloader):
#         outputs = model(**batch)
#         loss = outputs.loss
#         accelerator.backward(loss)

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)

#     # Evaluation
#     model.eval()
#     for step, batch in enumerate(eval_dataloader):
#         with torch.no_grad():
#             generated_tokens = accelerator.unwrap_model(model).generate(
#                 batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 max_new_tokens=1000,
#             )

#             generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
#             labels = batch["labels"]

#             # If we did not pad to max length, we need to pad the labels too
#             labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

#             generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
#             labels = accelerator.gather(labels).cpu().numpy()

#             # Replace -100 in the labels as we can't decode them
#             labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#             if isinstance(generated_tokens, tuple):
#                 generated_tokens = generated_tokens[0]
#             decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#             decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#             decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
#             # print('predictions={}, \nreferences={}'.format(decoded_preds, decoded_labels))
#             rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Compute metrics
    # result = rouge_score.compute()
    # print(result)
    # Extract the median ROUGE scores
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    # result = {k: round(v, 4) for k, v in result.items()}

    # scores = dict()
    # for name in result.keys():
    #     res = {k: round(v, 4) for k, v in result[name].items()}
    #     scores[name] = res

    # print(f"Epoch {epoch}:", scores)

    # # Save and upload
    # accelerator.wait_for_everyone()
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    # if accelerator.is_main_process:
    #     tokenizer.save_pretrained(output_dir)


class trainer:
    def __init__(self, args):
        # parameters
        self.data_path = args.data_path
        self.model_path = args.model_path
        self.output_dir = args.output_path
        self.runs = args.runs

        self.epoches = args.epoch
        self.batch_size = args.batchsize
        print('train parameters:{}'.format(args))
        # basic set
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

        self.train_dataloader, self.eval_dataloader = self.dataloader()

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)

        self.accelerator = Accelerator()
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.eval_dataloader)

        self.rouge_score = RougeScoreChinese()

        self.num_training_steps = self.epoches * len(self.train_dataloader)
        self.lr_scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps,)

        self.progress_bar = tqdm(range(self.num_training_steps), ncols=180)  # progress bar
        self.writer = SummaryWriter(self.runs)

    def run(self):
        scores_ = dict()
        for epoch in range(self.epoches):

            loss = self.train()
            self.eval()

            result = self.rouge_score.compute()
            
            scores = dict()
            for name in result.keys():
                res = {k: round(v, 4) for k, v in result[name].items()}
                scores[name] = res
            
            self.progress_bar.set_description('epoch:{},r1:{},r2:{},rl:{}'.format(epoch+1, 
            list(scores['rouge-1'].values()), list(scores['rouge-2'].values()),list(scores['rouge-l'].values())))
            self.progress_bar.refresh()

            self.write_rouge(scores, epoch)
            self.writer.add_scalar('loss', loss, epoch)
            # print(f"Epoch {epoch}:", scores)
            scores_[str(epoch)] = scores
            # Save and upload
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(self.output_dir, save_function=self.accelerator.save)
            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(self.output_dir)

        print(scores_)
        self.writer.close()

    def train(self):
        self.model.train()
        # progress_bar = tqdm(range(self.num_training_steps))
        loss_ = 0
        for step, batch in enumerate(self.train_dataloader):
            outputs = self.model(**batch)
            loss = outputs.loss
            loss_ += loss
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.progress_bar.update(1)
        return loss_/self.num_training_steps
    
    def eval(self):
        self.model.eval()
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=1000,
                )

                generated_tokens = self.accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                labels = batch["labels"]

                # If we did not pad to max length, we need to pad the labels too
                labels = self.accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id
                )

                generated_tokens = self.accelerator.gather(generated_tokens).cpu().numpy()
                labels = self.accelerator.gather(labels).cpu().numpy()

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                # print('predictions={}, \nreferences={}'.format(decoded_preds, decoded_labels))
                self.rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    def dataloader(self):
        dataset = DatasetDict.from_csv({'train': '{}/eval.csv'.format(self.data_path), 'validation': '{}/test.csv'.format(self.data_path)})
        tokenized_datasets = dataset.map(self.preprocess_function, batched=True)

        tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)  # 删除原本的列名

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        tokenized_datasets.set_format("torch")

        train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=self.batch_size)
        eval_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=self.batch_size)
        return train_dataloader, eval_dataloader
    
    def preprocess_function(self, examples):
        max_input_length = 512
        max_target_length = 30

        model_inputs = self.tokenizer(examples["text"],max_length=max_input_length,truncation=True,padding=True)
        labels = self.tokenizer(examples["target"], max_length=max_target_length, truncation=True,padding=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def write_rouge(self, rouge_dict, epoch):

        for name in rouge_dict.keys():
            for key in rouge_dict[name]:
                self.writer.add_scalar('{}/{}'.format(name, key), rouge_dict[name][key], epoch)


if "__main__" == __name__:
    parser = argparse.ArgumentParser(prog='Train', description='medical nlp adverse event report.', epilog='Copyright(r), 2024')
    parser.add_argument('-dp', '--data_path', required=False, default='./data/LCSTS_csv', type=str, help='data path')
    parser.add_argument('-mp', '--model_path', required=False, default="./hub/mt5-small-finetuned-on-mT5-lcsts", type=str, help='model path')
    parser.add_argument('--output_path', required=False, default="./results/mt5-finetuned-squad-accelerate", type=str, help='output path')
    parser.add_argument('--runs', required=False, default="./runs", type=str, help='output path')

    parser.add_argument('--epoch', required=False, default=3, type=int, help='epoches')
    parser.add_argument('--batchsize', required=False, default=64, type=int, help='batch size')

    args = parser.parse_args()

    trainer(args=args).run()



# tensorboard --logdir=/path/to/logs/ --port=xxxx