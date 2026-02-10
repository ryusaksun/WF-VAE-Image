CUDA_VISIBLE_DEVICES=0 python scripts/recon_single_image.py \
    --model_name WFVAE2Image \
    --model_config examples/wfvae2-image-1024.json \
    --ckpt_path /path/to/checkpoint-XXXX.ckpt \
    --image_path assets/gt_5544.jpg \
    --rec_path rec.jpg \
    --device cuda \
    --resolution 1024
