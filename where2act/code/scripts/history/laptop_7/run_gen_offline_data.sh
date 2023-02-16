python gen_offline_data.py \
  --data_dir ../data/laptop_7_pushing_train_7000 \
  --data_fn ../stats/laptop_7.txt \
  --category_types Laptop \
  --primact_types pushing \
  --num_processes 18 \
  --num_epochs 100 \
  --ins_cnt_fn ../stats/ins_cnt_laptop_7.txt \

python gen_offline_data.py \
  --data_dir ../data/laptop_7_pushing_validation_1400 \
  --data_fn ../stats/laptop_7.txt \
  --category_types Laptop \
  --primact_types pushing \
  --num_processes 18 \
  --num_epochs 20 \
  --ins_cnt_fn ../stats/ins_cnt_laptop_7.txt \

