if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Lead" ]; then
    mkdir ./logs/Lead
fi

sh scripts/backbone/Crossformer/Solar.sh
itr=1
seq_len=336
tag=_max
tau=1.0
data=Solar
model_name=Crossformer

for pred_len in 24
do
  learning_rate=0.001
  leader_num=4
  state_num=2
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --learning_rate $learning_rate > logs/Lead/$model_name'_pretrain_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 48
do
  learning_rate=0.0001
  leader_num=4
  state_num=8
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --pretrain --freeze \
    --learning_rate $learning_rate > logs/Lead/$model_name'_pretrain_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 96
do
  learning_rate=0.001
  leader_num=4
  state_num=8
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --pretrain --freeze \
    --learning_rate $learning_rate > logs/Lead/$model_name'_pretrain_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 192
do
  learning_rate=0.0001
  leader_num=2
  state_num=8
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --pretrain --freeze \
    --learning_rate $learning_rate > logs/Lead/$model_name'_pretrain_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 336
do
  learning_rate=0.00005
  leader_num=2
  state_num=8
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau \
    --pretrain --freeze \
    --learning_rate $learning_rate > logs/Lead/$model_name'_pretrain_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done