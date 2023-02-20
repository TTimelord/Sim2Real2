python train_3d_critic.py \
    --exp_suffix train_3d_critic_10211_train-val=1500-100_off-on=100-1 \
    --model_version model_3d_critic \
    --primact_type pushing \
    --category_types Laptop \
    --data_dir_prefix ../data/laptop_10211 \
    --offline_data_dir ../data/laptop_10211_pushing_train \
    --val_data_dir ../data/laptop_10211_pushing_validation \
    --val_data_fn data_tuple_list.txt \
    --train_shape_fn ../stats/laptop_10211.txt \
    --ins_cnt_fn ../stats/ins_cnt_laptop_10211.txt \
    --buffer_max_num 10000 \
    --num_processes_for_datagen 20 \
    --num_interaction_data_offline 100 \
    --num_interaction_data 1 \
    --sample_succ \
    --epochs 200 \
    --overwrite

