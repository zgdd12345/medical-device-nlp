import os, argparse, json
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

# Pegasus
from tokenizers_pegasus import PegasusTokenizer
from transformers import PegasusForConditionalGeneration

os.environ['TRANSFORMERS_OFFLINE'] = '1'
torch.cuda.empty_cache()


class trainer:
    def __init__(self, args):
        # parameters
        self.data_path = args.data_path
        self.model_path = model_path[args.model]

        self.output_dir = os.path.join(args.output_path, args.model)
        self.runs = os.path.join(args.runs, args.model)

        self.epoches = args.epoch
        self.batch_size = args.batchsize
        self.eval_batch_size = args.e_batchsize
        print('train parameters:{}'.format(args))
        # basic set
        if (args.model == 'pegasus_238') or (args.model == 'pegasus_523'):
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_path, use_fast=True)
            self.model = PegasusForConditionalGeneration.from_pretrained(self.model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

        self.train_dataloader, self.eval_dataloader = self.dataloader()

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)

        self.accelerator = Accelerator()
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.eval_dataloader)

        self.rouge_score = RougeScoreChinese()

        self.num_training_steps = self.epoches * len(self.train_dataloader)
        self.lr_scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps,)

        self.progress_bar = tqdm(range(self.num_training_steps + self.epoches * len(self.eval_dataloader)), ncols=180)  # progress bar
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
        self.save_json(scores_)
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
                if len(decoded_labels) != 0 and len(decoded_preds) != 0:
                    self.rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

                self.progress_bar.update(1)

    def dataloader(self):
        dataset = DatasetDict.from_csv({'train': '{}/train.csv'.format(self.data_path), 'validation': '{}/test.csv'.format(self.data_path)})
        tokenized_datasets = dataset.map(self.preprocess_function, batched=True)

        tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)  # 删除原本的列名

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        tokenized_datasets.set_format("torch")

        train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=self.batch_size)
        eval_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=self.eval_batch_size)
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

    def save_json(self,dic):
        with open(os.path.join(self.output_dir, 'score.json'), 'w') as json_file:
            json.dump(dic, json_file)


model_path = {
    'mt5':'./hub/mt5-small-finetuned-on-mT5-lcsts',
    'bert':'./hub/Bert-chinese-Summarization',
    'T5_base':'./hub/T5-base-Summarization',
    'T5_large':'./hub/T5-large-chinese-Summarization',
    'pegasus_238':'./hub/Randeng-Pegasus-238M-Summary-Chinese',
    'pegasus_523':'./hub/Randeng-Pegasus-523M-Summary-Chinese',
    'heackmt5':'./hub/HeackMT5-ZhSum100k',
}

if "__main__" == __name__:
    parser = argparse.ArgumentParser(prog='Train', description='medical device nlp adverse event report.', epilog='Copyright(r), 2024')
    parser.add_argument('--data_path',   required=False, default='./data/medical', type=str, help='data path')
    parser.add_argument('--model',       required=False, default="mt5", type=str, choices=model_path.keys(), help='model path')  # choice=model_path.keys(), 
    parser.add_argument('--output_path', required=False, default="./results", type=str, help='output path')
    parser.add_argument('--runs',        required=False, default="./runs", type=str, help='output path')

    parser.add_argument('--epoch',       required=False, default=1, type=int, help='epoches')
    parser.add_argument('--batchsize',   required=False, default=16, type=int, help='batch size')
    parser.add_argument('--e_batchsize', required=False, default=32, type=int, help='eval batch size')

    args = parser.parse_args()

    trainer(args=args).run()


# tensorboard --logdir=./runs

# python train.py --model mt5 --output_path ./results/mt5 --epoch 10 --runs ./runs/mt5 --batchsize 32
