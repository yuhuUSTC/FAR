### ---------------------------- FAR_Base ----------------------------  ###
# HF_ENDPOINT=https://hf-mirror.com torchrun --nnodes=1 --nproc_per_node=8  main_far.py \
# --img_size 256 --vae_path /video_hy2/modelzoo/yuhu_ckpt/pretrained/vae_mar/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model far_base --diffloss_d 6 --diffloss_w 1024 \
# --eval_bsz 32 --num_images 50000 \
# --num_iter 10 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
# --output_dir /ossfs/workspace/yuhu.yh/code/FAR/logs \
# --resume /video_hy2/modelzoo/yuhu_ckpt/pretrained/FAR \
# --data_path /mnt/workspace/workgroup/yuhu/data/imagenet-1k --evaluate --mask

### ---------------------------- FAR_Large ----------------------------  ###
HF_ENDPOINT=https://hf-mirror.com torchrun --nnodes=1 --nproc_per_node=1  main_far.py \
--img_size 256 --vae_path /video_hy2/modelzoo/yuhu_ckpt/pretrained/vae_mar/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model far_large --diffloss_d 3 --diffloss_w 1024 \
--eval_bsz 32 --num_images 1000 \
--num_iter 10 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
--output_dir /ossfs/workspace/yuhu.yh/code/FAR/logs \
--resume /video_hy2/modelzoo/yuhu_ckpt/pretrained/FAR \
--data_path /mnt/workspace/workgroup/yuhu/data/imagenet-1k --evaluate

### ---------------------------- FAR_Huge ----------------------------  ###
# HF_ENDPOINT=https://hf-mirror.com torchrun --nnodes=1 --nproc_per_node=8  main_far.py \
# --img_size 256 --vae_path /video_hy2/modelzoo/yuhu_ckpt/pretrained/vae_mar/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model far_huge --diffloss_d 3 --diffloss_w 1024 \
# --eval_bsz 32 --num_images 50000 \
# --num_iter 10 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
# --output_dir /ossfs/workspace/yuhu.yh/code/FAR/logs \
# --resume /video_hy2/modelzoo/yuhu_ckpt/pretrained/FAR \
# --data_path /mnt/workspace/workgroup/yuhu/data/imagenet-1k --evaluate --mask

