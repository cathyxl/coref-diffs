echo "It is a train"
exp_name="coref-diff"
echo $exp_name
# for WoW
data_dir="data/wow"
out_dir="models/wow"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 graphks_main.py \
--dataset wow \
--epochs 10 \
--encoder_out_dim 320 \
--gat_hid_dim 1024 \
--train_batch_size 4 \
--eval_batch_size 4 \
--lr 0.00001 \
--exp_name $out_dir \
--data_dir $data_dir \
--bert_config bert-base-uncased \