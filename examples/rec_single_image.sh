CUDA_VISIBLE_DEVICES=0 python scripts/recon_single_image.py \
    --model_name WFVAE2Image \
    --from_pretrained /path/to/checkpoint_or_hf_model \
    --image_path assets/gt_5544.jpg \
    --rec_path rec.jpg \
    --device cuda \
    --resolution 1024
