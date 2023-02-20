python gen_offline_data.py \
  --data_dir ../data/drawer_45677_pulling_train_9000 \
  --data_fn ../stats/drawer_45677.txt \
  --category_types Drawer \
  --primact_types pulling \
  --num_processes 20 \
  --num_epochs 900 \
  --ins_cnt_fn ../stats/ins_cnt_drawer_45677.txt \

python gen_offline_data.py \
  --data_dir ../data/drawer_45677_pulling_validation_1500 \
  --data_fn ../stats/drawer_45677.txt \
  --category_types Drawer \
  --primact_types pulling \
  --num_processes 20 \
  --num_epochs 150 \
  --ins_cnt_fn ../stats/ins_cnt_drawer_45677.txt \

