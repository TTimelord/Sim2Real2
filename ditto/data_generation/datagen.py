"""
    Batch-generate data
"""

import os
import numpy as np
import multiprocessing as mp
from subprocess import call
from utils import printout
import time
from tqdm import tqdm


class DataGen(object):

    def __init__(self, num_processes, flog=None):
        self.num_processes = num_processes
        self.flog = flog
        
        self.todos = []
        self.processes = []
        self.is_running = False
        self.Q = mp.Queue()

    def __len__(self):
        return len(self.todos)

    def add_one_collect_job(self, shape_id, category, cnt_id, data_dir, stereo_data_dir):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('COLLECT', shape_id, category, cnt_id, data_dir, stereo_data_dir, np.random.randint(10000000))
        self.todos.append(todo)
    
    @staticmethod
    def job_func(pid, todos):
        succ_todos = []
        for todo in tqdm(todos):
            if todo[0] == 'COLLECT':
                cmd = 'python data_generation/collect_data.py %s %s %d --out_dir %s --stereo_out_dir %s --random_seed %d --no_gui > /dev/null 2>&1' \
                        % (todo[1], todo[2], todo[3], todo[4], todo[5], todo[6])
                # print(cmd)
            ret = call(cmd, shell=True)

    def start_all(self):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot start all while DataGen is running!')
            exit(1)

        total_todos = len(self)
        num_todos_per_process = int(np.ceil(total_todos / self.num_processes))
        np.random.shuffle(self.todos)
        for i in range(self.num_processes):
            todos = self.todos[i*num_todos_per_process: min(total_todos, (i+1)*num_todos_per_process)]
            p = mp.Process(target=self.job_func, args=(i, todos))
            p.start()
            self.processes.append(p)
            # p.join(5)
            # p.terminate()
        
        self.is_running = True


