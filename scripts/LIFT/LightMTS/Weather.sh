if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Lead" ]; then
    mkdir ./logs/Lead
fi

itr=5
seq_len=336
tau=1.0
data=Weather
model_name=LightMTS

learning_rate=0.001
for pred_len in 24 48
do
  leader_num=8
  state_num=12
  python -u run_longExp.py \
    --dataset $data \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr $itr \
    --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate > logs/Lead/$model_name'_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 96
do
  leader_num=8
  state_num=8
  python -u run_longExp.py \
    --dataset $data \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr $itr \
    --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate > logs/Lead/$model_name'_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 192
do
  leader_num=4
  state_num=8
  python -u run_longExp.py \
    --dataset $data \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr $itr \
    --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate > logs/Lead/$model_name'_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 336 720
do
  leader_num=2
  state_num=8
  python -u run_longExp.py \
    --dataset $data \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr $itr \
    --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate > logs/Lead/$model_name'_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done