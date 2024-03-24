if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/backbone" ]; then
    mkdir ./logs/backbone
fi

itr=1
seq_len=336
data=ECL
model_name=PatchTST
train_epochs=100
for pct_start in 0.2
do
for pred_len in 24 48 96 192 336 720
do
for learning_rate in 0.0001
do
  python -u run_longExp.py \
    --dataset $data --model $model_name --seq_len $seq_len --pred_len $pred_len \
    --itr $itr \
    --train_epochs $train_epochs \
    --pct_start $pct_start \
    --patience 10 \
    --learning_rate $learning_rate > logs/backbone/$model_name'_'$data'_'$pred_len'_lr'$learning_rate'_epoch'$train_epochs'_pct'$pct_start.log 2>&1
done
done
done