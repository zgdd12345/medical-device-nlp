import os
from tqdm import tqdm

import jieba
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ['TRANSFORMERS_OFFLINE'] = '1'

app = FastAPI()


class RiskWarning:
    def __init__(self, request) -> None:
        self.request = request
        self.data = request.text
        self.model = request.model
        
        # self.source_model = source_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.risk_sets = {
            '维修', '断裂', '故障', '破损', '粘连', '无法', '更换', '重新', '延误', '导致', 
            '漏尿', '瘙痒', '不能', '包装', '折弯', '漏液', '开机', '脱落', '漏气', '破裂', 
            '弯曲', '停用', '堵塞', '异常', '严重', '损坏', '损伤', '拔出', '异物', '污染', 
            '裂痕', '没有', '漏血', '蓝屏', '报错', '缺少', '不通', '折痕', '过热', '突然', 
            '下落', '报警', '换下', '烧坏', '渗出', '裂隙', '碎裂', '粘性', '不严', '滴出', 
            '漏水', '不易', '流出', '报修', '有断', '连接', '裂缝', '不出', '不亮', '弯裂', 
            '痕迹', '不停', '倒刺', '过敏', '变慢', '不滴', '分离', '开裂', '不亮', '脱管', 
            '孔漏', '损耗', '不足', '发烫', '烧糊', '发暗', '失败', '过高', '松弛', '老化', 
            '严重', '所致', '频繁', '漏电', '困难', '密封', '反复', '碎片', '缺损', '困难', 
            '松动', '死机', '关机', '重启', '断开', '阻塞', '碎屑', '毛刺', '刺伤', '解离', 
            '下降', '略感', '不适', '穿刺', '杂质', '外露', '出血', '歪斜', '黏贴', '不紧',
            '并未', '脱出', '松紧', '关闭', '错误', '不转', '不准', '自检', '不过', '杂乱',
            '粘贴', '漏血', '补充', '残缺', '变黑', '浪费', '换药', '不转', '破洞', '外渗',
            '缺失', '敷贴', '胶痕', '发红', '泄漏', '过快', '黑屏', '生锈', '爆裂', '去除',
            '通电', '不满', '涂抹', '闪烁', '不全', '敷料', '皮肤', '掉落', '不牢', '滴水',
            '敷贴', '受限', '异响', '尖锐', '高温', '掉落', '溢出', '外漏', '卡纸', '不开',
            '缝隙', '未封', '晃动', '灰色', '脏块', '自行', '剧烈', '明显', '减轻', '干扰',
            '发黑', '轻微', '裂纹', '裂口', '不好', '漏药', '无气', '不够', '噪音', '过大',
            '固定', '充电', '提前', '受阻', '变扁', 

            '连接处', '抽不出', '保险丝', '不灵敏', '松紧带', '中不热', '灯不亮', '显示屏', '充不上', '打不开',
            '充不进', '看不清',

            '系统故障', '接触不良',

            '碎', '无', '漏', '不', '卷', '粘', '松', '血', '少', '新',
            '少', '未', '换', '新', '贴', '敷', '断', '掉', '缺',
            }

        self.model_path = {
            'mt5':'./results/mt5',
            'bert':'./results/bert',
            # 'T5_base':'./results/T5_base',
            'T5_large':'./results/T5_large',
            # 'pegasus_238':'./hub/Randeng-Pegasus-238M-Summary-Chinese',
            # 'pegasus_523':'./hub/Randeng-Pegasus-523M-Summary-Chinese',
            # 'heackmt5':'./hub/HeackMT5-ZhSum100k',
        }

    def run(self):
        risk_words = self.get_risk_words()
        # self._save_csv(out_summary=False, out_words=True)
        risk_summary = self.inference()
        # self._save_csv(out_summary=True, out_words=True)
        return {'summary':risk_summary, 'words':risk_words,}
 
    def inference(self):
        # self.source_model = self.model_path[self.model]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path[self.model], cache_dir = self.model_path[self.model], use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path[self.model], cache_dir = self.model_path[self.model]).to(self.device)

        res = []
        for text in tqdm(self.data, ncols=100):
            input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            summary_ids = self.model.generate(input_ids,
                                        min_length=15,
                                        max_length=60,
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
            # print(summary_text)
            res.append(summary_text)
        return res

    def get_risk_words(self):
        # filter_word = {
        #             '，', '。',  '_', '.', ')', '(','-', ' ', '：', ':', '（', '）', '、',
        #             '2023', '1', '2', '3', '4', '5', '6', '7', '8', '9', '2022', '2024',
        #             '年', '月', '日',
        #             }
        
        risk_words_list = []
        for text in self.data:
            # text = self.data[num]['event']
            # print(text)
            # part_words = jieba.lcut(cont)
            part_words = jieba.lcut_for_search(text)
            # part_words = [word_ for word_ in part_words if word_ not in filter_word]

            risk_words = set()
            for word in part_words:
                if word in self.risk_sets:
                    risk_words.add(word)
            risk_words_list.append(risk_words)

        return risk_words_list

    # def _read_riskdict(self):
    #     words = set()
    #     with open(self.riskdict_path, 'r') as f:
    #         for word in f.readlines():
    #             words.add(word.strip())
    #     return words


    # def _read_csv(self):
    #     events = dict()
    #     # for sheet in ['1月', '2月', '3月', '4月', '5', '6']:
    #     df = pd.read_excel(self.data_path, sheet_name='{}'.format('Sheet0'))
    #     for i in df.index:
    #         events.update({str(df['报告编码'][i]):{'event':df['使用过程'][i]}})
    #     return events

    # def _save_csv(self, out_summary=False, out_words=True):
    #     dic = {}
    #     dic.update({'risk_summary':[]}) if out_summary else None
    #     dic.update({'risk_words':[]}) if out_words else None

    #     for num in self.data.keys():
    #         # dic['num'].append(num)
    #         if out_summary:
    #             dic['risk_summary'].append(self.data[num]['risk_summary'])
    #         if out_words:
    #             dic['risk_words'].append(self.data[num]['risk_words'])

    #     df = pd.DataFrame(dic, index=self.data.keys())
    #     df.to_csv(self.csv_save_path)


class TextGenerationRequest(BaseModel):
    text: list
    model: str = 'mt5'

    min_length: int = 15
    max_length: int = 60
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

@app.post("/generate")
def generate_risk_warning(request:TextGenerationRequest):

    try:
        # riskwarning = RiskWarning(request, source_model).inference()
        riskwarning = RiskWarning(request).run()
        print(riskwarning)
        return {"generated_text": riskwarning}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/items/{item_id}")
# async def read_item(item_id):
#     return {"item_id": item_id}


model_path = {
    'mt5':'./results/mt5',
    'bert':'./results/bert',
    # 'T5_base':'./results/T5_base',
    'T5_large':'./results/T5_large',
    # 'pegasus_238':'./hub/Randeng-Pegasus-238M-Summary-Chinese',
    # 'pegasus_523':'./hub/Randeng-Pegasus-523M-Summary-Chinese',
    # 'heackmt5':'./hub/HeackMT5-ZhSum100k',
}

if '__main__' == __name__:
    uvicorn.run(app, host='0.0.0.0', port=11234)

    # csv_path = './data/2023.xls'
    # riskdict_path = './data/riskwords.txt'
    # source_model = "./hub/mt5-small-finetuned-on-mT5-lcsts"
    # source_model = "./hub/T5-base-Summarization"
    # csv_save_path = './data/test_res.csv'
    # request = {"text":["2023.4.14患者进行气管插管的口腔护理时，第一次使用止血钳进行拧干棉球操作，突然中间断裂无法使用，不能对棉球拧干吗，立即更换护理包，继续为患者进行口腔护理，此次事件未对患者造成损害",
    #                    "2023年4月8日，科室在使用注射泵过程中，发现设备停止工作。科室立刻停止使用，报修给设备科，设备维修人员检查发现设备供电接口碎裂故障，导致设备插头无法固定供电不稳，维修接口后修复。对患者的注射治疗造成延误。",
    #                    "2023.04.18患者郭京梅因腹痛收入我院，护士在给患者进行静脉输液时，给患者进行留置针正压接头端消毒，发现回弹头无法回弹，接上输液器时碘伏能流进留置针里，立即更换新的留置针，未对患者造成伤害。",
    #                    "2023年4月16日，患者因龋齿来院就诊，医生在给患者做显微根管治疗术时，发现灯泡不亮了，联系设备科。",
    #                    "2023.1.6护理在为患者输液准备时，发现输液器茂菲滴管里面的阀弹不上去，立即更换，未对患者造成伤害",
    #                    "患者因皮脂腺感染所致肿物于2023年3月30日入院，入院进行切除后进行包扎，打开凡士林纱布时发现其与包装袋黏连，随即更换新的油纱。",
    #                    ]}
    # for key in model_path:
    #     source_model = model_path[key]
    #     riskwarning = RiskWarning(request, source_model='./results/mt5').inference()
    #     print('model:{}, summary:{}'.format(key, riskwarning))
    # riskwarning = RiskWarning(csv_path, source_model, csv_save_path).run()
# 2023.4.14患者进行气管插管的口腔护理时，第一次使用止血钳进行拧干棉球操作，突然中间断裂无法使用，不能对棉球拧干吗，立即更换护理包，继续为患者进行口腔护理，此次事件未对患者造成损害

# uvicorn main:app --host 0.0.0.0 --port 11234

# curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"text":["2023.4.14患者进行气管插管的口腔护理时，第一次使用止血钳进行拧干棉球操作，突然中间断裂无法使用，不能对棉球拧干吗，立即更换护理包，继续为患者进行口腔护理，此次事件未对患者造成损害","2023年4月8日，科室在使用注射泵过程中，发现设备停止工作。科室立刻停止使用，报修给设备科，设备维修人员检查发现设备供电接口碎裂故障，导致设备插头无法固定供电不稳，维修接口后修复。对患者的注射治疗造成延误。","2023.04.18患者郭京梅因腹痛收入我院，护士在给患者进行静脉输液时，给患者进行留置针正压接头端消毒，发现回弹头无法回弹，接上输液器时碘伏能流进留置针里，立即更换新的留置针，未对患者造成伤害。","2023年4月16日，患者因龋齿来院就诊，医生在给患者做显微根管治疗术时，发现灯泡不亮了，联系设备科。","2023.1.6护理在为患者输液准备时，发现输液器茂菲滴管里面的阀弹不上去，立即更换，未对患者造成伤害","患者因皮脂腺感染所致肿物于2023年3月30日入院，入院进行切除后进行包扎，打开凡士林纱布时发现其与包装袋黏连，随即更换新的油纱。",]}'
# 
# curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"text": ["2023.4.14患者进行气管插管的口腔护理时，第一次使用止血钳进行拧干棉球操作，突然中间断裂无法使用，不能对棉球拧干吗，立即更换护理包，继续为患者进行口腔护理，此次事件未对患者造成损害",]}'
# curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"text": ["2023.4.14患者进行气管插管的口腔护理时，第一次使用止血钳进行拧干棉球操作，突然中间断裂无法使用，不能对棉球拧干吗，立即更换护理包，继续为患者进行口腔护理，此次事件未对患者造成损害"], "max_length": 100, "temperature": 0.7}'
# curl -X POST http://www.arclighttest.cn:11234/generate -H "Content-Type: application/json" -d '{"text": ["2023.4.14患者进行气管插管的口腔护理时，第一次使用止血钳进行拧干棉球操作，突然中间断裂无法使用，不能对棉球拧干吗，立即更换护理包，继续为患者进行口腔护理，此次事件未对患者造成损害","2023年4月8日，科室在使用注射泵过程中，发现设备停止工作。科室立刻停止使用，报修给设备科，设备维修人员检查发现设备供电接口碎裂故障，导致设备插头无法固定供电不稳，维修接口后修复。对患者的注射治疗造成延误。","2023.04.18患者郭京梅因腹痛收入我院，护士在给患者进行静脉输液时，给患者进行留置针正压接头端消毒，发现回弹头无法回弹，接上输液器时碘伏能流进留置针里，立即更换新的留置针，未对患者造成伤害。","2023年4月16日，患者因龋齿来院就诊，医生在给患者做显微根管治疗术时，发现灯泡不亮了，联系设备科。","2023.1.6护理在为患者输液准备时，发现输液器茂菲滴管里面的阀弹不上去，立即更换，未对患者造成伤害","患者因皮脂腺感染所致肿物于2023年3月30日入院，入院进行切除后进行包扎，打开凡士林纱布时发现其与包装袋黏连，随即更换新的油纱。"]}'
