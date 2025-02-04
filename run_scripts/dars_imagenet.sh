#!/usr/bin/env bash
cd ..

sigma=$1
steps=$2
budget_jump_to_guiding_ratio=$3
seed=$4

python eval_certified_densepure.py \
--exp logs/friday \
--config imagenet.yml \
-i scale_sig$sigma-ratio$budget_jump_to_guiding_ratio \
--domain imagenet \
--seed $seed \
--diffusion_type guided-ddpm \
--lp_norm L2 \
--budget_jump_to_guiding_ratio $budget_jump_to_guiding_ratio \
--outfile logs/imagenet/dars_sig$sigma-ratio$budget_jump_to_guiding_ratio/seed$seed/certify \
--sigma $sigma \
--N 10000 \
--N0 100 \
--certified_batch 120 \
--sample_id $(seq -s ' ' 0 1000 49000) \
--use_id \
--certify_mode purify \
--advanced_classifier beit \
--use_t_steps \
--num_t_steps $steps \
--reverse_seed 1 \
--scale 5
