########## exp setup ##########
export centers_num=3
dataset_name=chickchicken_qwen
clip_feat_dim=3
clip_feat_name=clip_features_dim${clip_feat_dim}
qwen_feat_dim=3
qwen_feat_name=qwen_features_dim${qwen_feat_dim}

export use_discrete_lang_f=f
export use_tribute_dlang=f
level=0
clip_level=1

base_exp=hypernerf/${dataset_name}/${dataset_name}_${level}_base
clip_exp=hypernerf/${dataset_name}/${dataset_name}_${level}_clipF${clip_feat_dim}
qwen_exp=hypernerf/${dataset_name}/${dataset_name}_${level}_qwenF${qwen_feat_dim}

# Small iterations for quick testing
# base_coarse_iters=3000
# base_fine_iters=10000
# clip_lang_iters=30
# qwen_lang_iters=30

# Full-training defaults (from train.py/arguments):
base_coarse_iters=3000      # maps to --coarse_base_iterations
base_fine_iters=10000       # maps to --fine_base_iterations
clip_lang_iters=10000       # maps to --fine_lang_iterations (CLIP)
qwen_lang_iters=10000       # maps to --fine_lang_iterations (Qwen)

# ########## 1) Base splat with RGB only (no language stages) ##########
# echo "===== Base splat with RGB only (no language stages)"
# python train.py -s data/hypernerf/${dataset_name} --port 6021 --expname ${base_exp} --configs arguments/hypernerf/chicken.py \
#   --coarse_base_iterations ${base_coarse_iters} --coarse_lang_iterations 0 --fine_base_iterations ${base_fine_iters} --fine_lang_iterations 0 \
#   --feature_level ${level} --save_iterations ${base_fine_iters}

# ########## Helper: copy base checkpoint into language experiment dirs ##########
# latest_base_ckpt_dir=$(ls -d output/${base_exp}/point_cloud/fine-base_iteration_* 2>/dev/null | sort -V | tail -n1)
# if [ -z "${latest_base_ckpt_dir}" ]; then
#   echo "Error: no fine-base checkpoint found in output/${base_exp}." >&2
#   exit 1
# fi

# mkdir -p output/${clip_exp}/point_cloud
# mkdir -p output/${qwen_exp}/point_cloud
# rsync -a ${latest_base_ckpt_dir}/ output/${clip_exp}/point_cloud/$(basename ${latest_base_ckpt_dir})/
# rsync -a ${latest_base_ckpt_dir}/ output/${qwen_exp}/point_cloud/$(basename ${latest_base_ckpt_dir})/

# ########## 2) Train language features (CLIP) on frozen geometry ##########
# echo "===== Train language features (CLIP) on frozen geometry"
# export language_feature_hiddendim=${clip_feat_dim}
# python train.py -s data/hypernerf/${dataset_name} --port 6021 --expname ${clip_exp} --configs arguments/hypernerf/chicken.py --include_feature \
#   --language_features_name ${clip_feat_name} --feature_level ${clip_level} \
#   --coarse_base_iterations 0 --coarse_lang_iterations 0 --fine_base_iterations 0 --fine_lang_iterations ${clip_lang_iters} \
#   --no_dlang 1 --resume_from_stage fine-base --resume_from_iter -1 \
#   --save_iterations ${clip_lang_iters}

# # for mode in "lang" "rgb"; do
# for mode in "lang"; do
#   echo "===== Render ${mode} images for CLIP"
#   python render.py -s data/hypernerf/${dataset_name} --language_features_name ${clip_feat_name} --model_path output/${clip_exp} \
#     --feature_level ${level} --skip_train --skip_test --configs arguments/hypernerf/chicken.py --mode ${mode} --no_dlang 1 --load_stage fine-lang
# done

# ########## 3) Train language features (Qwen) on the exact same frozen geometry ##########
# echo "===== Train language features (Qwen) on the exact same frozen geometry"
# export language_feature_hiddendim=${qwen_feat_dim}
# python train.py -s data/hypernerf/${dataset_name} --port 6021 --expname ${qwen_exp} --configs arguments/hypernerf/chicken.py --include_feature \
#   --language_features_name ${qwen_feat_name} --feature_level ${level} \
#   --coarse_base_iterations 0 --coarse_lang_iterations 0 --fine_base_iterations 0 --fine_lang_iterations ${qwen_lang_iters} \
#   --no_dlang 1 --resume_from_stage fine-base --resume_from_iter -1 \
#   --save_iterations ${qwen_lang_iters}

# # for mode in "lang" "rgb"; do
# for mode in "lang"; do
#   echo "===== Render ${mode} images for Qwen"
#   python render.py -s data/hypernerf/${dataset_name} --language_features_name ${qwen_feat_name} --model_path output/${qwen_exp} \
#     --feature_level ${level} --skip_train --skip_test --configs arguments/hypernerf/chicken.py --mode ${mode} --no_dlang 1 --load_stage fine-lang
# done

######### 4) Extract graph ##########
echo "===== Extract graph"
# Ensure Python-side modules size to CLIP F when loading the CLIP model
export language_feature_hiddendim=${clip_feat_dim}
python extract_graph.py -s data/hypernerf/${dataset_name} --language_features_name ${clip_feat_name} --model_path output/${clip_exp} \
  --feature_level ${level} --skip_train --skip_test --configs arguments/hypernerf/chicken.py --mode lang --no_dlang 1 --load_stage fine-lang \
  --num_views 5 --clip_autoencoder_ckpt_path autoencoder/ckpt/${dataset_name}_clip/best_ckpt.pth \
  --qwen_autoencoder_ckpt_path autoencoder/ckpt/${dataset_name}_qwen/best_ckpt.pth \
  --rgb_model_path output/${base_exp} --rgb_load_stage fine-base \
  --clip_model_path output/${clip_exp} --clip_load_stage fine-lang \
  --qwen_model_path output/${qwen_exp} --qwen_load_stage fine-lang