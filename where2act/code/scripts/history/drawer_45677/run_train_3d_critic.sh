python train_3d_critic.py \
    --exp_suffix train_3d_critic_45677_train-val=9000-1500_off-on=800-1 \
    --model_version model_3d_critic \
    --primact_type pulling \
    --category_types Drawer \
    --data_dir_prefix ../data/drawer_45677 \
    --offline_data_dir ../data/drawer_45677_pulling_train_9000 \
    --val_data_dir ../data/drawer_45677_pulling_validation_1500 \
    --val_data_fn data_tuple_list.txt \
    --train_shape_fn ../stats/drawer_45677.txt \
    --ins_cnt_fn ../stats/ins_cnt_drawer_45677.txt \
    --buffer_max_num 10000 \
    --num_processes_for_datagen 20 \
    --num_interaction_data_offline 800 \
    --num_interaction_data 1 \
    --sample_succ \
    --epochs 100 \
    --overwrite

