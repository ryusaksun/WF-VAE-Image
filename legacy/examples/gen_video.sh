# Legacy video reconstruction (archived)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

REAL_VIDEO_DIR=/path/to/eval/videos
EXP_NAME=wfvae2-video-recon
CKPT=/path/to/legacy_video_checkpoint
SUBSET_SIZE=0

accelerate launch \
    --config_file examples/accelerate_configs/default_config.yaml \
    legacy/scripts/recon_video.py \
    --real_video_dir ${REAL_VIDEO_DIR} \
    --generated_video_dir video_gen/${EXP_NAME} \
    --from_pretrained ${CKPT} \
    --model_name WFVAE2 \
    --resolution 336 \
    --num_frames 17 \
    --sample_rate 1 \
    --batch_size 1 \
    --num_workers 8 \
    --subset_size ${SUBSET_SIZE} \
    --output_origin
