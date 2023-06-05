python gen_offline_data.py \
  --data_dir ../data/drawer_35_pushing_train_350_3 \
  --data_fn ../stats/drawer_35.txt \
  --category_types Drawer \
  --primact_types pushing \
  --num_processes 16 \
  --num_epochs 1 \
  --ins_cnt_fn ../stats/ins_cnt_drawer_35.txt \


