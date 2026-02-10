unset https_proxy
export WANDB_PROJECT=WFVAE-IMAGE
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

EXP_NAME=WFVAE2IMAGE-1024

# Update these paths before running.
TRAIN_IMAGE_DIR=/path/to/train/images
EVAL_IMAGE_DIR=/path/to/eval/images

torchrun \
    --nnodes=1 --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=12135 \
    train_image_ddp.py \
    --exp_name ${EXP_NAME} \
    --image_path ${TRAIN_IMAGE_DIR} \
    --eval_image_path ${EVAL_IMAGE_DIR} \
    --model_name WFVAE2Image \
    --model_config examples/wfvae2-image-1024.json \
    --resolution 1024 \
    --eval_resolution 1024 \
    --eval_crop_size 1024 \
    --batch_size 4 \
    --grad_accum_steps 8 \
    --lr 0.00001 \
    --epochs 10 \
    --save_ckpt_step 500 \
    --eval_steps 200 \
    --eval_batch_size 2 \
    --eval_subset_size 64 \
    --disc_start 1000 \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 1.0 \
    --loss_type l1
