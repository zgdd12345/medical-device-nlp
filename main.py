import os
from tqdm import tqdm

import jieba
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ['TRANSFORMERS_OFFLINE'] = '1'


class RiskWarning:
    def __init__(self, data_path, riskdict_path, source_model, csv_save_path) -> None:
        self.data_path = data_path
        self.riskdict_path = riskdict_path
        self.source_model = source_model
        self.csv_save_path = csv_save_path

        self.data = self._read_csv()

    def run(self):
        risk_words = self.get_risk_words()
        self._save_csv(out_summary=False, out_words=True)
        risk_summary = self.inference()
        self._save_csv(out_summary=True, out_words=True)
        return self.data
 
    def inference(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.source_model, cache_dir = self.source_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.source_model, cache_dir = self.source_model)

        for num in tqdm(self.data.keys(), ncols=100):
            text = self.data[num]['event']
            input_ids = self.tokenizer.encode(text, return_tensors='pt')
            summary_ids = self.model.generate(input_ids,
                                        min_length=5,
                                        max_length=50,
                                        num_beams=10,
                                        repetition_penalty=2.5,
                                        length_penalty=1.0,
                                        early_stopping=True,
                                        no_repeat_ngram_size=2,
                                        use_cache=True,
                                        do_sample = True,
                                        temperature = 0.8,
                                        top_k = 50,
                                        top_p = 0.95)
            
            summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            self.data[num].update({'risk_summary':summary_text})

        return self.data

    def get_risk_words(self):
        risk_dict = self._read_riskdict()
        # self.data = self.read_csv()

        filter_word = {
                    '，', '。',  '_', '.', ')', '(','-', ' ', '：', ':', '（', '）', '、',
                    '2023', '1', '2', '3', '4', '5', '6', '7', '8', '9', '2022', 
                    '年', '月', '日',
                    }

        for num in self.data.keys():
            text = self.data[num]['event']
            # print(text)
            # part_words = jieba.lcut(cont)
            part_words = jieba.lcut_for_search(text)
            part_words = [word_ for word_ in part_words if word_ not in filter_word]

            risk_words = set()
            for word in part_words:
                if word in risk_dict:
                    risk_words.add(word)
            self.data[num].update({'risk_words': risk_words})
            # if risk_words:
            #     print('risk word:{}, \nsentence:{}\n'.format(risk_words, text))
            #     for i in range(0,len(part_words), 20):
            #         print('participle{}:{}'.format((i+20)//20,part_words[i:i+20]))
            #     print('\n')
        return self.data

    def _read_riskdict(self):
        words = set()
        with open(self.riskdict_path, 'r') as f:
            for word in f.readlines():
                words.add(word.strip())
        return words

    def _read_csv(self):
        events = dict()
        # for sheet in ['1月', '2月', '3月', '4月', '5', '6']:
        df = pd.read_excel(self.data_path, sheet_name='{}'.format('Sheet0'))
        for i in df.index:
            events.update({str(df['报告编码'][i]):{'event':df['使用过程'][i]}})
        return events

    def _save_csv(self, out_summary=False, out_words=True):
        dic = {}
        dic.update({'risk_summary':[]}) if out_summary else None
        dic.update({'risk_words':[]}) if out_words else None

        for num in self.data.keys():
            # dic['num'].append(num)
            if out_summary:
                dic['risk_summary'].append(self.data[num]['risk_summary'])
            if out_words:
                dic['risk_words'].append(self.data[num]['risk_words'])

        df = pd.DataFrame(dic, index=self.data.keys())
        df.to_csv(self.csv_save_path)

if '__main__' == __name__:
    csv_path = './data/2023.xls'
    riskdict_path = './data/riskwords.txt'
    source_model = "./hub/mt5-small-finetuned-on-mT5-lcsts"
    csv_save_path = './data/test_res.csv'

    riskwarning = RiskWarning(csv_path, riskdict_path, source_model, csv_save_path).run()
