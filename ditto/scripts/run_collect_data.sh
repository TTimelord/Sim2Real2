# python data_generation/collect_data.py \
#   0007 \
#   Drawer \
#   1 \
# python data_generation/collect_data.py \
#   44817 \
#   Drawer \
#   1 \

# python data_generation/collect_data.py \
#   857 \
#   Faucet \
#   1 \

# python data_generation/collect_data.py \
#   9748 \
#   Laptop \
#   1 \

# python data_generation/collect_data.py \
#   10211 \
#   Laptop \
#   1 \

# python data_generation/collect_data.py 0008 Drawer 3 --out_dir data/laptop_train  # > /dev/null 2>&1
# python data_generation/collect_data.py 10626 Laptop 0 --out_dir data/laptop_train  # > /dev/null 2>&1
python data_generation/collect_data.py 0011 Faucet 0 --out_dir results --stereo_out_dir results/stereo # > /dev/null 2>&1