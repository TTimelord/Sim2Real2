python gen_offline_data.py \
  --data_dir ../data/laptop_10211_pushing_train \
  --data_fn ../stats/laptop_10211.txt \
  --category_types Laptop \
  --primact_types pushing \
  --num_processes 20 \
  --num_epochs 150 \
  --ins_cnt_fn ../stats/ins_cnt_laptop_10211.txt \

python gen_offline_data.py \
  --data_dir ../data/laptop_10211_pushing_validation \
  --data_fn ../stats/laptop_10211.txt \
  --category_types Laptop \
  --primact_types pushing \
  --num_processes 20 \
  --num_epochs 10 \
  --ins_cnt_fn ../stats/ins_cnt_laptop_10211.txt \

