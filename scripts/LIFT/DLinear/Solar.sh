if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Lead" ]; then
    mkdir ./logs/Lead
fi

itr=5
seq_len=336
tau=1.0
data=Solar
model_name=DLinear

learning_rate=0.005
for pred_len in 24 48
do
  leader_num=4
  state_num=16
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate > logs/Lead/$model_name'_LIFT_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 96
do
  leader_num=4
  state_num=12
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate > logs/Lead/$model_name'_LIFT_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 192 336
do
  leader_num=8
  state_num=16
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate > logs/Lead/$model_name'_LIFT_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 720
do
  leader_num=8
  state_num=16
  learning_rate=0.001
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate > logs/Lead/$model_name'_LIFT_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done