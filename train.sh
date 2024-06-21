# python train.py --model mt5         --epoch 10 --batchsize 16 --e_batchsize 32
# python train.py --model bert        --epoch 10 --batchsize 16 --e_batchsize 32
# python train.py --model T5_base     --epoch 10 --batchsize 32 --e_batchsize 64
# python train.py --model T5_large    --epoch 10 --batchsize 2 --e_batchsize 2
# python train.py --model pegasus_238 --epoch 10 --batchsize 32 --e_batchsize 64
# python train.py --model pegasus_523 --epoch 10 --batchsize 8 --e_batchsize 16
python train.py --model heackmt5    --epoch 10 --batchsize 4 --e_batchsize 4
