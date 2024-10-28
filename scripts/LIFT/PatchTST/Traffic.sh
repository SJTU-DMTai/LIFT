if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Lead" ]; then
    mkdir ./logs/Lead
fi

itr=1
seq_len=336
tag=_max
tau=1.0
data=Traffic
model_name=PatchTST
train_epochs=100
state_num=1
for pred_len in 24
do
  pct_start=0.2
  leader_num=4
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --patience 10 --pct_start $pct_start --train_epochs $train_epochs \
    --learning_rate $learning_rate > logs/Lead/$model_name'_LIFT_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate'_epoch'$train_epochs'_pct'$pct_start.log 2>&1
done
for pred_len in 48 336
do
  pct_start=0.2
  leader_num=4
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --patience 10 --pct_start $pct_start \
    --learning_rate $learning_rate > logs/Lead/$model_name'_LIFT_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate'_epoch'$train_epochs'_pct'$pct_start.log 2>&1
done
for pred_len in 96
do
  pct_start=0.3
  leader_num=8
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --patience 10 --pct_start $pct_start \
    --learning_rate $learning_rate > logs/Lead/$model_name'_LIFT_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate'_epoch'$train_epochs'_pct'$pct_start.log 2>&1
done
for pred_len in 192
do
  pct_start=0.3
  leader_num=8
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --patience 10 --pct_start $pct_start --train_epochs $train_epochs \
    --learning_rate $learning_rate > logs/Lead/$model_name'_LIFT_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate'_epoch'$train_epochs'_pct'$pct_start.log 2>&1
done
for pred_len in 720
do
  pct_start=0.1
  leader_num=4
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --patience 10 --pct_start $pct_start --train_epochs $train_epochs \
    --learning_rate $learning_rate > logs/Lead/$model_name'_LIFT_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate'_epoch'$train_epochs'_pct'$pct_start.log 2>&1
done
