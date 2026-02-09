export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

REAL_IMAGE_DIR=/path/to/eval/images
EXP_NAME=wfvae2image-recon
CKPT=/path/to/checkpoint_or_hf_model
SUBSET_SIZE=0

accelerate launch \
    --config_file examples/accelerate_configs/default_config.yaml \
    scripts/recon_image.py \
    --real_image_dir ${REAL_IMAGE_DIR} \
    --generated_image_dir image_gen/${EXP_NAME} \
    --from_pretrained ${CKPT} \
    --model_name WFVAE2Image \
    --resolution 1024 \
    --crop_size 1024 \
    --batch_size 2 \
    --num_workers 8 \
    --subset_size ${SUBSET_SIZE} \
    --output_origin
