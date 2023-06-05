import os
import sys
from argparse import ArgumentParser

from datagen import DataGen

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--stereo_data_dir', type=str, help='stereo_depth data directory')
parser.add_argument('--data_fn', type=str, help='data file that indexs all shape-ids')
parser.add_argument('--category_types', type=str,
                    help='list all categories [separated by comma], default: None, meaning all', default=None)
parser.add_argument('--num_processes', type=int, default=40, help='number of CPU cores to use')
parser.add_argument('--ins_cnt_fn', type=str,
                    help='a file listing all category instance count, which is used to balance the interaction data '
                        'amount to make sure that all categories have roughly same amount of data interaction, ' 
                        'regardless of different shape counts in these categories')
conf = parser.parse_args()

if not os.path.exists(conf.data_dir):
    os.makedirs(conf.data_dir)
if not os.path.exists(conf.stereo_data_dir):
    os.makedirs(conf.stereo_data_dir)

conf.category_types = conf.category_types.split(',')
print(conf.category_types)

cat2freq = dict()
with open(conf.ins_cnt_fn, 'r') as fin:
    for l in fin.readlines():
        cat, _, freq = l.rstrip().split()
        cat2freq[cat] = int(freq)
print(cat2freq)

datagen = DataGen(conf.num_processes)

with open(conf.data_fn, 'r') as fin:
    for l in fin.readlines():
        shape_id, cat = l.rstrip().split()
        if cat in conf.category_types:
            for cnt_id in range(cat2freq[cat]):
                # print(shape_id, cat, epoch, cnt_id)
                datagen.add_one_collect_job(shape_id, cat, cnt_id, conf.data_dir, conf.stereo_data_dir)

datagen.start_all()
