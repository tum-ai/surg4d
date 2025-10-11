cd autoencoder
dataset_name=chickchicken_qwen
dataset_path=../data/hypernerf/${dataset_name}
feature_name=qwen_features
latent_dim=3
echo "Training Qwen autoencoder (3584 -> ${latent_dim})"

python train_qwen.py --dataset_path ${dataset_path} --model_name ${dataset_name}_qwen \
    --language_name ${feature_name} --epochs 100 --lr 1e-4 --batch_size 256 --latent_dim ${latent_dim}

python test_qwen.py --dataset_path ${dataset_path} --model_name ${dataset_name}_qwen \
    --language_name ${feature_name} --latent_dim ${latent_dim}