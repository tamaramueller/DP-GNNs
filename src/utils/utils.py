import random
import numpy as np
import torch
import logging
import os


def initialize_logging(name_log_file:str, path_to_log_folder:str="../logging/", log_level=logging.INFO, file_mode="w"):
    logging.basicConfig(filename=os.path.join(path_to_log_folder, name_log_file), encoding='utf-8', level=log_level, file_mode=file_mode)


def make_deterministic(SEED=13):
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cpu_generator = torch.Generator(device="cpu").manual_seed(SEED)
    # gpu_generator = torch.Generator(device="cuda").manual_seed(SEED)
    
    return cpu_generator, None# gpu_generator


def convert_label_to_indicator_array(label, nr_classes):
    vec = np.array([0] * nr_classes)
    vec[label] = 1

    return vec
