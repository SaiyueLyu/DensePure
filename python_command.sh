python /mnt/home/DensePure/eval_certified_densepure.py \
--exp /mnt/home/DensePure/logs/imagenet \
--config /mnt/home/DensePure/configs/imagenet.yml \
-i dars_sig1-ratio1 \
--domain imagenet \
--seed 0 \
--diffusion_type guided-ddpm \
--lp_norm L2 \
--budget_jump_to_guiding_ratio 1 \
--outfile /mnt/home/DensePure/logs/imagenet/dars_sig1-ratio1/seed0/certify \
--sigma 1 \
--N 10000 \
--N0 100 \
--certified_batch 120 \
--sample_id $(seq -s ' ' 0 1000 49000) \
--use_id \
--certify_mode purify \
--advanced_classifier beit \
--use_t_steps \
--num_t_steps 10 \
--reverse_seed 1


# eai job submit : be sure to change sigma seed ratio!!!!!!!!!!!!
eai job new --preemptable --cpu 4 --gpu 1 --gpu-mem 80 --gpu-model-filter a100 --mem 16 -d snow.colab_public.data:/mnt/colab_public:rw -e HOME=/home/toolkit -d snow.home.saiyue_lyu:/mnt/home:rw -i registry.console.elementai.com/snow.interactive_toolkit/saiyue -- /mnt/home/denp/bin/python /mnt/home/DensePure/eval_certified_densepure.py --exp /mnt/home/DensePure/logs/test --config /mnt/home/DensePure/configs/imagenet.yml -i test --domain imagenet --seed 0 --diffusion_type guided-ddpm --lp_norm L2 --budget_jump_to_guiding_ratio 1 --outfile /mnt/home/DensePure/logs/test --sigma 1 --N 10000 --N0 100 --certified_batch 120 --sample_id $(seq -s ' ' 0 1000 49000) --use_id --certify_mode purify --advanced_classifier beit --use_t_steps --num_t_steps 10 --reverse_seed 1 --scale 7



eai job new --preemptable --cpu 4 --gpu 1 --gpu-mem 80 --gpu-model-filter a100 --mem 16 -d snow.colab_public.data:/mnt/colab_public:rw -e HOME=/home/toolkit -d snow.home.saiyue_lyu:/mnt/home:rw -i registry.console.elementai.com/snow.interactive_toolkit/saiyue -- /mnt/home/denp/bin/python /mnt/home/DensePure/certify.py --toolkit
