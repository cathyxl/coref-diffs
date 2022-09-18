echo "It is a test"
echo "test wow best"
data_dir="data/wow"
out_dir="models/wow"

# for WoW
CUDA_VISIBLE_DEVICES=0 python3 graphks_test.py \
--dataset wow \
--resume $out_dir/model_best.pth.tar \
--encoder_out_dim 320 \
--gat_hid_dim 1024 \
--eval_batch_size 8 \
--exp_name $out_dir \
--data_dir $data_dir \
--bert_config bert-base-uncased \