if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Lead" ]; then
    mkdir ./logs/Lead
fi

sh scripts/backbone/MTGNN/ECL.sh
itr=1
seq_len=336
tag=_max
tau=0.1
data=ECL
model_name=MTGNN

for pred_len in 24 48 96
do
  leader_num=8; state_num=8; learning_rate=0.01
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau --pretrain --freeze \
    --learning_rate $learning_rate > logs/Lead/$model_name'_pretrain_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 192
do
  leader_num=8; state_num=2; learning_rate=0.005
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau --pretrain --freeze \
    --learning_rate $learning_rate > logs/Lead/$model_name'_pretrain_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 336
do
  leader_num=8; state_num=2; learning_rate=0.01
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau --pretrain --freeze \
    --learning_rate $learning_rate > logs/Lead/$model_name'_pretrain_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done
for pred_len in 720
do
  leader_num=8; state_num=4; learning_rate=0.005
  python -u run_longExp.py \
    --dataset $data --model $model_name --lift --seq_len $seq_len --pred_len $pred_len --itr $itr --checkpoints "" \
    --leader_num $leader_num --state_num $state_num --temperature $tau --pretrain --freeze \
    --learning_rate $learning_rate > logs/Lead/$model_name'_pretrain_'$data'_'$pred_len'_K'$leader_num'_tau'$tau'_state'$state_num'_lr'$learning_rate.log 2>&1
done