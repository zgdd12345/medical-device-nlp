# medical report nlp

## Requirements
    datasets==2.20.0
    huggingface-hub==0.23.4
    jieba==0.42.1
    nltk==3.9.1
    numpy==1.26.0
    pandas==2.2.2
    rouge==1.0.1
    rouge-chinese==1.0.3
    tokenizers==0.19.1
    torch==2.3.1
    torchaudio==2.3.1
    torchvision==0.18.1
    tornado==6.4.1
    tqdm==4.66.4
    transformers==4.42.3



## Data Preparation

You should split your data into train and test data

    data
    └───train.csv
    └───test.csv


## Training

### Training Help

    usage: Train [-h] [--data_path DATA_PATH]
                [--model {mt5,bert,T5_base,T5_large,pegasus_238,pegasus_523,heackmt5}]
                [--output_path OUTPUT_PATH] [--runs RUNS] [--epoch EPOCH]
                [--batchsize BATCHSIZE] [--e_batchsize E_BATCHSIZE]

    options:
    -h, --help            show this help message and exit
    --data_path DATA_PATH
                            data path
    --model {mt5,bert,T5_base,T5_large,pegasus_238,pegasus_523,heackmt5}
                            model path
    --output_path OUTPUT_PATH
                            output path
    --runs RUNS           output path
    --epoch EPOCH         epoches
    --batchsize BATCHSIZE
                            batch size
    --e_batchsize E_BATCHSIZE
                            eval batch size


### Sample Usage
    python train.py --model mt5 --output_path ./results/mt5 --epoch 10 --runs ./runs/mt5 --batchsize 32

or you can write a shell scripts


### Pre-training model

You can download the pre-training model weight from huggingface
