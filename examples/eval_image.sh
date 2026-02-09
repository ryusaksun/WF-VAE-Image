EXP_NAME=wfvae2image-recon
REAL_IMAGE_DIR=image_gen/${EXP_NAME}/origin
GENERATED_IMAGE_DIR=image_gen/${EXP_NAME}
BATCH_SIZE=4

for METRIC in lpips psnr ssim; do
  echo "Evaluating ${METRIC} ..."
  accelerate launch \
      --config_file examples/accelerate_configs/default_config.yaml \
      scripts/eval_image.py \
      --batch_size ${BATCH_SIZE} \
      --real_image_dir ${REAL_IMAGE_DIR} \
      --generated_image_dir ${GENERATED_IMAGE_DIR} \
      --resolution 1024 \
      --crop_size 1024 \
      --metric ${METRIC}
done
