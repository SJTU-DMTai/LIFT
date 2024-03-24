if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/backbone" ]; then
    mkdir ./logs/backbone
fi

itr=1
seq_len=336
data=PeMSD8
model_name=PatchTST
train_epochs=100
pct_start=0.2
for pred_len in 24 48 96 192 336 720
do
for learning_rate in 0.001
do
  python run_longExp.py \
    --dataset $data --batch_size 24 \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr $itr \
    --train_epochs $train_epochs \
    --pct_start $pct_start \
    --patience 10 \
    --learning_rate $learning_rate > logs/backbone/$model_name'_'$data'_'$pred_len'_lr'$learning_rate'_epoch'$train_epochs'_pct'$pct_start.log 2>&1
done
done