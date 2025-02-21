#!/usr/bin/env bash

for ind in {2..49}; do
    eai job new --preemptable --cpu 4 --gpu 1 --gpu-mem 32 --gpu-model-filter v100 --mem 16 \
    -d snow.colab_public.data:/mnt/colab_public:rw -e HOME=/home/toolkit -d snow.home.saiyue_lyu:/mnt/home:rw \
    -i registry.console.elementai.com/snow.interactive_toolkit/saiyue \
    -- /mnt/home/denp/bin/python /mnt/home/DensePure/certify.py \
    --toolkit --v100 --id_index $ind
done
