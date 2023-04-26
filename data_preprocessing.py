from pathlib import Path
import shutil
import os

# MICC-F2000
# 1300 -> original
# 700 -> forged
# 2000 -> total

base_dir = "/home/brechtl/Pictures/Data/MICC/MICC-F2000/"

Path(base_dir + "original").mkdir(parents=True, exist_ok=True)
Path(base_dir + "forged").mkdir(parents=True, exist_ok=True)

with open(base_dir + "groundtruthDB_2000.txt") as GT_file:
    for line in GT_file:
        img_name, gt = line.split()
        if os.path.exists(base_dir + img_name):
            if gt == '0':
                shutil.move(base_dir + img_name, base_dir + "original/" + img_name)
            else:
                shutil.move(base_dir + img_name, base_dir + "forged/" + img_name)
