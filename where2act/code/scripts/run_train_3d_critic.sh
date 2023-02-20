python train_3d_critic.py \
    --exp_suffix faucet_8_critic_train-val=48000-9600_off-on=500-1 \
    --model_version model_3d_critic \
    --primact_type pushing-left \
    --category_types Faucet \
    --data_dir_prefix ../data/faucet_8 \
    --offline_data_dir ../data/faucet_8_pushing-left_train_48000 \
    --val_data_dir ../data/faucet_8_pushing-left_validation_9600 \
    --val_data_fn data_tuple_list.txt \
    --train_shape_fn ../stats/faucet_8.txt \
    --ins_cnt_fn ../stats/ins_cnt_faucet_8.txt \
    --buffer_max_num 10000 \
    --num_processes_for_datagen 20 \
    --num_interaction_data_offline 500 \
    --num_interaction_data 1 \
    --sample_succ \
    --epochs 100 \
    --overwrite \
    --num_point_per_shape 2000 \
    --abs_thres 0.17 \
    --rel_thres 0.1 \
    # abs thres 10 degree


