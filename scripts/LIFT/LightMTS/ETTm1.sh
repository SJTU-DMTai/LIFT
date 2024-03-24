if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Lead" ]; then
    mkdir ./logs/Lead
fi

itr=5
seq_len=336
tau=1.0
data=ETTm1
model_name=LightMTS
leader_num=1
state_num=2

for pred_len in 24
do
  learning_rate=0.001
  python -u run_longExp.py \
    --dataset $data \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr $itr \
    --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --tag "_naive" \
    --learning_rate $learning_rate > logs/Lead/$model_name'_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 48 96
do
  learning_rate=0.005
  python -u run_longExp.py \
    --dataset $data \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr $itr \
    --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --tag "_naive" \
    --learning_rate $learning_rate > logs/Lead/$model_name'_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 192 720
do
  learning_rate=0.01
  python -u run_longExp.py \
    --dataset $data \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr $itr \
    --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --tag "_naive" \
    --learning_rate $learning_rate > logs/Lead/$model_name'_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 336
do
  state_num=1; learning_rate=0.005
  python -u run_longExp.py \
    --dataset $data \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr $itr \
    --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --tag "_naive" \
    --learning_rate $learning_rate > logs/Lead/$model_name'_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done