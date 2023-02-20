python train_3d_critic.py \
    --exp_suffix laptop_7_critic_train-val=7000-1400_off-on=90-1 \
    --model_version model_3d_critic \
    --primact_type pushing \
    --category_types Laptop \
    --data_dir_prefix ../data/laptop_7 \
    --offline_data_dir ../data/laptop_7_pushing_train_7000 \
    --val_data_dir ../data/laptop_7_pushing_validation_1400 \
    --val_data_fn data_tuple_list.txt \
    --train_shape_fn ../stats/laptop_7.txt \
    --ins_cnt_fn ../stats/ins_cnt_laptop_7.txt \
    --buffer_max_num 10000 \
    --num_processes_for_datagen 20 \
    --num_interaction_data_offline 90 \
    --num_interaction_data 1 \
    --sample_succ \
    --epochs 100 \
    --overwrite \
    --num_point_per_shape 2000 \


