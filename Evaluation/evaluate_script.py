
import pandas as pd
import numpy as np
import re
from copy import deepcopy
from collections import defaultdict
file_path = "./evaluate_dk.out"
versions = ['SEQUENTIAL', "DATA-OPENMP", "FEATURE", 'CUDA', 'MPI', 'DATA-FEATURE', "NODE"]
baseline = 'SEQUENTIAL'

all_data = defaultdict(list)

def get_data(model):
    p = re.compile("^(.+) (.+): (.+)")
    data = []
    with open(file_path, "r") as f:
        new_data = {"MODEL": model}
        for line in f.readlines():
            model_name = line.strip().split(" ")[0]
            if model_name != model:
                continue
            finds = re.findall(p, line)
            if len(finds) == 0 or len(finds[0]) != 3:
                continue
            var_name = finds[0][1].strip()
            value = finds[0][2].strip()
            if var_name == 'DATASET':
                if len(new_data) > 1:
                    data.append(new_data)
                new_data = {"MODEL": model}
                
            new_data[var_name] = value
        if len(new_data) > 1:
            data.append(new_data)
    return data

def cal_speedup(data1, data2):
    for d in data1:
        dataset = d['DATASET']
        for d2 in data2:
            if d2['DATASET'] == dataset:
                d['TRAIN_SPEEDUP'] = float(d2['Train_Time']) / float(d['Train_Time'])
                d['COMPRESS_SPEEDUP'] = np.nan if float(d['COMPRESS_TIME']) == 0 else float(d2['COMPRESS_TIME']) / float(d['COMPRESS_TIME'])
                d['SPLIT_SPEEDUP'] = np.nan if float(d['SPLIT_TIME']) == 0 else float(d2['SPLIT_TIME']) / float(d['SPLIT_TIME'])
    return data1

SEQUENTIAL = get_data(baseline)
MPI = get_data("MPI")
MPI = cal_speedup(MPI, SEQUENTIAL)
DATA_FEATURE = get_data("DATA-FEATURE")
DATA_FEATURE = cal_speedup(DATA_FEATURE, SEQUENTIAL)
NODE = get_data("NODE")
NODE = cal_speedup(NODE, SEQUENTIAL)


pd.DataFrame(MPI+SEQUENTIAL+DATA_FEATURE+NODE).to_excel("./evaluation_result_general.xlsx", index=False)





        
        