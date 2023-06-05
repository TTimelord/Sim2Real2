python gen_offline_data.py \
  --data_dir ../data/drawer_35_pushing_train_70000 \
  --data_fn ../stats/drawer_35.txt \
  --category_types Drawer \
  --primact_types pushing \
  --num_processes 15 \
  --num_epochs 200 \
  --ins_cnt_fn ../stats/ins_cnt_drawer_35.txt \

python gen_offline_data.py \
  --data_dir ../data/drawer_35_pushing_validation_14000 \
  --data_fn ../stats/drawer_35.txt \
  --category_types Drawer \
  --primact_types pushing \
  --num_processes 5 \
  --num_epochs 40 \
  --ins_cnt_fn ../stats/ins_cnt_drawer_35.txt \

