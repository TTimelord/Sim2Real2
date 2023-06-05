python gen_offline_data.py \
  --data_dir /data/where2act/drawer_35_pushing_train_77000 \
  --data_fn ../stats/drawer_35.txt \
  --category_types Drawer \
  --primact_types pushing \
  --num_processes 15 \
  --num_epochs 220 \
  --ins_cnt_fn ../stats/ins_cnt_drawer_35.txt \

python gen_offline_data.py \
  --data_dir /data/where2act/drawer_35_pushing_validation_14000 \
  --data_fn ../stats/drawer_35.txt \
  --category_types Drawer \
  --primact_types pushing \
  --num_processes 15 \
  --num_epochs 40 \
  --ins_cnt_fn ../stats/ins_cnt_drawer_35.txt \

