# from nltk import sent_tokenize
from rouge_chinese import Rouge
import jieba, nltk


class RougeScoreChinese:
    def __init__(self) -> None:
        self.hyp = []
        self.ref = []

        self.rouge = Rouge()

    def add_batch(self, predictions, references):
        predictions = ' '.join(jieba.cut(predictions)) 
        references = ' '.join(jieba.cut(references)) 
        self.hyp.append(predictions)
        self.ref.append(references)

    def compute(self):
        return self.rouge.get_scores(self.hyp, self.ref, avg=True)
    

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds[0], labels[0]
