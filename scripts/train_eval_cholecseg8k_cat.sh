########## exp setup ##########
export centers_num=3
clip_feat_dim=6
# video_feat_dim=6
video_name=video27
clip_name=video27_00480
dataset_path=data/cholecseg8k/preprocessed_ssg/${video_name}/${clip_name}
language_feature_name=qwen_cat_features_dim6
# dataset_name=video01_28820_final_for_training_monst3r_pcd
# dataset_path=data/cholecseg8k/${dataset_name}
# language_feature_name=language_features_3d

########## time-agnostic language field ##########
export language_feature_hiddendim=${clip_feat_dim}
# rm -rf submodules/4d-langsplat-rasterization/build && pip install --no-cache-dir -e submodules/4d-langsplat-rasterization
export use_discrete_lang_f=f
python train.py -s  ${dataset_path} --port 6021 --expname cholecseg8k/${clip_name}_qwen_cat --configs arguments/cholecseg8k/default.py --include_feature \
    --language_features_name ${language_feature_name} --feature_level 0 --joint_coarse --no_dlang 1 --no_ds --depth_loss_weight 1.0
for mode in "lang" "rgb"; do
python render.py -s  ${dataset_path} --language_features_name ${language_feature_name} --model_path output/cholecseg8k/${clip_name}_qwen_cat \
    --feature_level 0 --skip_train --skip_test --configs arguments/cholecseg8k/default.py --mode ${mode} --no_dlang 1 --load_stage fine-lang --no_ds
done

######### 4) Extract graph ##########
echo "===== Extract graph"
# Ensure Python-side modules size to CLIP F when loading the CLIP model
export language_feature_hiddendim=${clip_feat_dim}
python extract_graph.py \
  -s  ${dataset_path} \
  --language_features_name ${language_feature_name} \
  --model_path output/cholecseg8k/${clip_name}_qwen_cat \
  --feature_level 0 --skip_train --skip_test \
  --configs arguments/cholecseg8k/default.py --mode lang --no_dlang 1 --load_stage fine-lang --no_ds \
  --num_views 5 \
  --qwen_autoencoder_ckpt_path data/cholecseg8k/preprocessed_ssg/${video_name}/${clip_name}/autoencoder/best_ckpt.pth \
  --store_verbose # store features of filtered gaussians etc., turn off when running with whole dataset

python cluster.py -s  ${dataset_path} --language_features_name ${language_feature_name} --model_path output/cholecseg8k/${dataset_name} \
    --feature_level 0 --skip_train --skip_test --configs arguments/cholecseg8k/default.py --mode lang --no_dlang 1 --load_stage fine-lang --num_views 5 --autoencoder_ckpt_path autoencoder/ckpt/${dataset_name}_clip/best_ckpt.pth

########## time-sensitive language field ##########
# level=0
# language_feature_name=video_features
# export language_feature_hiddendim=${video_feat_dim}
# rm -rf submodules/4d-langsplat-rasterization/build 
# pip install --no-cache-dir -e submodules/4d-langsplat-rasterization
# export use_discrete_lang_f=f
# python train.py -s data/hypernerf/${dataset_name} --port 6021 --expname hypernerf/${dataset_name}/${dataset_name}_${level} --configs arguments/hypernerf/chicken.py --include_feature \
#     --language_features_name ${language_feature_name}-language_features_dim${video_feat_dim} --feature_level ${level} --fine_lang_iterations 0 --joint_coarse --no_dlang 0 --checkpoint_iterations 10000

# export use_discrete_lang_f=t
# python train.py -s data/hypernerf/${dataset_name} --port 6021 --expname hypernerf/${dataset_name}/${dataset_name}_${level} --configs arguments/hypernerf/chicken.py --include_feature \
#     --language_features_name ${language_feature_name}-language_features_dim${video_feat_dim} --feature_level ${level} --joint_coarse --no_dlang 0 --resume_from_final_stage 1 --start_checkpoint output/hypernerf/${dataset_name}/${dataset_name}_${level}/chkpnt_fine-base_10000.pth

# for mode in "lang" "rgb"; do
# python render.py -s  data/hypernerf/${dataset_name} --feature_level ${level} --language_features_name ${language_feature_name}-language_features_dim${video_feat_dim} \
#     --model_path output/hypernerf/${dataset_name}/${dataset_name}_${level} --skip_train --skip_test --configs arguments/hypernerf/chicken.py --mode ${mode} --no_dlang 0 --load_stage fine-lang-discrete 
# done

# ########## Evaluate ##########
# cd eval
# python eval.py --dataset_type hypernerf \
#     --annotation_folder ../data/hypernerf/${dataset_name}/annotations \
#     --exp_name ${dataset_name}/${dataset_name} \
#     --feat_dim ${clip_feat_dim} \
#     --video_feat_dim ${video_feat_dim} \
#     --iterations 10000 \
#     --video_eval_iterations 20000 \
#     --ae_ckpt_path ../autoencoder/ckpt/${dataset_name}_clip/best_ckpt.pth \
#     --video_ae_ckpt_path ../autoencoder/ckpt/${dataset_name}_video/best_ckpt.pth \
#     --video_feat_dir ${dataset_name}/${dataset_name} \
#     --apply_video_search \
#     --smooth_feature_post