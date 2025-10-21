########## exp setup ##########
export centers_num=3
clip_feat_dim=6
# video_feat_dim=6
video_name=video27
clip_name=video27_00480
dataset_path=data/cholecseg8k/preprocessed_ssg/${video_name}/${clip_name}
language_feature_name=qwen_cat_features_dim6
exp_name=cholecseg8k/${clip_name}_qwen_cat_depth_opacity

########## time-agnostic language field ##########
export language_feature_hiddendim=${clip_feat_dim}
# rm -rf submodules/4d-langsplat-rasterization/build && pip install --no-cache-dir -e submodules/4d-langsplat-rasterization
export use_discrete_lang_f=f
# Training/rendering disabled (model already trained)
# python train.py -s  ${dataset_path} --port 6021 --expname ${exp_name} --configs arguments/cholecseg8k/no_tv.py --include_feature \
#     --language_features_name ${language_feature_name} --feature_level 0 --joint_coarse --no_dlang 1 --no_ds --depth_loss_weight 1.0 --opacity_loss_weight 1.0
# for mode in "lang" "rgb"; do
# python render.py -s  ${dataset_path} --language_features_name ${language_feature_name} --model_path output/${exp_name} \
#     --feature_level 0 --skip_train --skip_test --configs arguments/cholecseg8k/no_tv.py --mode ${mode} --no_dlang 1 --load_stage fine-lang
# done

######### 4) Extract graph ##########
echo "===== Extract graph (extract only, no configs)"
# Ensure Python-side modules size to CLIP F when loading the CLIP model
export language_feature_hiddendim=${clip_feat_dim}
/home/students/.pixi/bin/pixi run -q python extract_graph.py \
  -s  ${dataset_path} \
  --language_features_name ${language_feature_name} \
  --model_path output/${exp_name} \
  --feature_level 0 --skip_train --skip_test \
  --configs arguments/cholecseg8k/no_tv.py --mode lang --no_dlang 1 --load_stage fine-lang \
  --qwen_autoencoder_ckpt_path data/cholecseg8k/preprocessed_ssg/${video_name}/${clip_name}/autoencoder/best_ckpt.pth \
  --store_verbose # store features of filtered gaussians etc., turn off when running with whole dataset


