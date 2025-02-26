
### ---------------------------- FAR_Base ----------------------------  ###
# torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
# main_far.py \
# --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model far_base --diffloss_d 6 --diffloss_w 1024 \
# --epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 \
# --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
# --data_path ${IMAGENET_PATH}

### ---------------------------- FAR_Large ----------------------------  ###
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_far.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model far_large --diffloss_d 3 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH}

### ---------------------------- FAR_Huge ----------------------------  ###
# torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
# main_far.py \
# --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model far_huge --diffloss_d 3 --diffloss_w 1024 \
# --epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 \
# --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
# --data_path ${IMAGENET_PATH}
