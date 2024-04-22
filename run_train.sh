CUDA_VISIBLE_DEVICES=0 python train.py \
    --input_wavs_dir ./  \
    --input_training_file ./dataset/train_files.txt   \
    --input_validation_file ./dataset/test_files.txt \
    --config config_v1.json   \
    --checkpoint_path ./checkpoints_denoiser_24khz_13-04-2024
