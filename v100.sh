#!/bin/bash

for ind in {0..1}; do
    eai job new --preemptable --cpu 4 --gpu 1 --gpu-mem 32 --gpu-model-filter v100 --mem 16 -d snow.colab_public.data:/mnt/colab_public:rw -e HOME=/home/toolkit -d snow.home.saiyue_lyu:/mnt/home:rw -i registry.console.elementai.com/snow.interactive_toolkit/saiyue \
    -- /mnt/home/denp/bin/python /mnt/home/DensePure/eval_certified_densepure.py \
    --exp /mnt/home/DensePure/exp/imagenet/$ind \
    --config /mnt/home/DensePure/configs/imagenet.yml \
    -i /mnt/home/DensePure/denp_1.41421356237 \
    --domain imagenet \
    --seed 0 \
    --diffusion_type guided-ddpm \
    --lp_norm L2 \
    --outfile /mnt/home/DensePure/v100/1.41421356237_$ind \
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
    --id_index $ind
done
