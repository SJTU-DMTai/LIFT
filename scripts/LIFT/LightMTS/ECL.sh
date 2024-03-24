if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Lead" ]; then
    mkdir ./logs/Lead
fi

itr=5
seq_len=336
tau=0.1
data=ECL
model_name=LightMTS

for pred_len in 24
do
  learning_rate=0.005
  leader_num=8
  state_num=20
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
for pred_len in 48
do
  learning_rate=0.01
  leader_num=8
  state_num=16
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
  learning_rate=0.01
  leader_num=8
  state_num=20
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
  learning_rate=0.005
  leader_num=8
  state_num=2
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
for pred_len in 336
do
  learning_rate=0.01
  leader_num=4
  state_num=2
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
for pred_len in 720
do
  learning_rate=0.005
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