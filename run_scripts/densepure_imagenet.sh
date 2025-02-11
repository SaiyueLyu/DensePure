#!/usr/bin/env bash
cd ..

# sigma=$1
# steps=$2
# reverse_seed=$3
index=$1


python eval_certified_densepure.py \
--exp exp/imagenet \
--config configs/imagenet.yml \
-i denp_1.41421356237 \
--domain imagenet \
--seed 0 \
--diffusion_type guided-ddpm \
--lp_norm L2 \
--outfile v100/1.41421356237_$index \
--sigma 1.41421356237 \
--N 10000 \
--N0 100 \
--certified_batch 20 \
--sample_id $(seq -s ' ' 0 1000 49000) \
--use_id \
--certify_mode purify \
--advanced_classifier beit \
--use_t_steps \
--num_t_steps 10 \
--reverse_seed 1 \
--id_index $index

#--save_predictions \
#--predictions_path exp/imagenet/ \
