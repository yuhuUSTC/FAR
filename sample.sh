### ---------------------------- FAR_Base ----------------------------  ###
# torchrun --nnodes=1 --nproc_per_node=1  main_far.py \
# --img_size 256 --vae_path pretrained/vae_mar/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model far_base --diffloss_d 6 --diffloss_w 1024 \
# --eval_bsz 32 --num_images 1000 \
# --num_iter 10 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
# --output_dir pretrained_models/far/far_base \
# --resume pretrained_models/far/far_base \
# --data_path ${IMAGENET_PATH} --evaluate

### ---------------------------- FAR_Large ----------------------------  ###
torchrun --nnodes=1 --nproc_per_node=1  main_far.py \
--img_size 256 --vae_path pretrained/vae_mar/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model far_large --diffloss_d 3 --diffloss_w 1024 \
--eval_bsz 32 --num_images 1000 \
--num_iter 10 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
--output_dir pretrained_models/far/far_large \
--resume pretrained_models/far/far_large \
--data_path ${IMAGENET_PATH} --evaluate

### ---------------------------- FAR_Huge ----------------------------  ###
# torchrun --nnodes=1 --nproc_per_node=1  main_far.py \
# --img_size 256 --vae_path pretrained/vae_mar/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model far_huge --diffloss_d 3 --diffloss_w 1024 \
# --eval_bsz 32 --num_images 1000 \
# --num_iter 10 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
# --output_dir pretrained_models/far/far_huge \
# --resume pretrained_models/far/far_huge \
# --data_path ${IMAGENET_PATH} --evaluate

