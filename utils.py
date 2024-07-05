import os
import shutil
from config import *

def DATA_ORGANIZER(df,dir):


    file_label_pairs = df[['id', 'species']].values


    for id, species in file_label_pairs:

        label_dir = os.path.join(dir, species)
        os.makedirs(label_dir, exist_ok=True)
    
        id = f"{id}.jpg"
        src_path = os.path.join(dir, id)
        dst_path = os.path.join(label_dir, id)
        shutil.move(src_path, dst_path)


class COMPILE():
    LOSS = "mse"
    METRICS = ["accuracy"]
    OPTIMIZERS = "adam"