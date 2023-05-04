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


# MICC-F220
# 110 -> original
# 110 -> forged
# 220 -> total

base_dir = "/home/brechtl/Pictures/Data/MICC/MICC-F220/"

Path(base_dir + "original").mkdir(parents=True, exist_ok=True)
Path(base_dir + "forged").mkdir(parents=True, exist_ok=True)

with open(base_dir + "groundtruthDB_220.txt") as GT_file:
    for line in GT_file:
        img_name, gt = line.split()
        if os.path.exists(base_dir + img_name):
            if gt == '0':
                shutil.move(base_dir + img_name, base_dir + "original/" + img_name)
            else:
                shutil.move(base_dir + img_name, base_dir + "forged/" + img_name)


# CoMoFoD
# 200 -> original
# 199 -> forged
# 399 -> total

base_dir = "/home/brechtl/Pictures/Data/CoMoFoD_small_v2/"

Path(base_dir + "original").mkdir(parents=True, exist_ok=True)
Path(base_dir + "forged").mkdir(parents=True, exist_ok=True)

for filename in os.listdir(base_dir):
    if filename.endswith("_O.png"):
        shutil.move(base_dir + filename, base_dir + "original/" + filename)
    if filename.endswith("_F.png"):
        shutil.move(base_dir + filename, base_dir + "forged/" + filename)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        os.remove(base_dir + filename)


# CASIAv1
# 800 -> original
# 450 -> forged
# 1250 -> total

# Author made mistakes in file names so I removed them
# Found on https://github.com/namtpham/casia1groundtruth
to_remove = ["Sp_D_NND_A_nat0054_nat0054_0189.jpg", "Sp_D_NNN_A_cha0052_cha0052_0309.jpg",
             "Sp_D_NNN_A_cha0065_cha0065_0110.jpg", "Sp_D_NNN_A_nat0052_nat0052_0325.jpg",
             "Sp_D_NNN_A_pla0002_pla0002_0333.jpg", "Sp_D_NNN_A_pla0030_pla0030_0113.jpg",
             "Sp_D_NNN_R_arc0088_arc0088_0367.jpg", "Sp_D_NRD_A_pla0075_pla0075_0588.jpg",
             "Sp_D_NRN_A_cha0076_cha0076_0501.jpg", "Sp_S_CRN_A_arc0007_arc0005_0228.jpg",
             "Sp_S_NND_A_art0025_art0019_0090.jpg", "Sp_S_NND_A_art0050_art0080_0092.jpg",
             "Sp_S_NNN_A_arc0002_arc0026_0201.jpg", "Sp_S_NNN_A_arc0051_arc0029_0018.jpg",
             "Sp_S_NNN_A_sec0042_sec0046_0040.jpg", "Sp_S_NNN_R_arc0067_arc0071_0062.jpg",
             "Sp_S_NRD_A_ani0040_ani0054.0257.jpg", "Sp_S_NRD_A_art0003_art0042_0088.jpg",
             "Sp_S_NRN_A_arc0003_arc0009_0233.jpg", "Sp_S_NRN_A_arc0008_arc0028_0234.jpg",
             "Sp_S_NRN_A_arc0036_arc0033_0236.jpg", "Sp_S_NRN_A_arc0038_arc0039_0237.jpg",
             "Sp_S_NRN_A_arc0048_arc0050_0238.jpg", "Sp_S_NRN_A_arc0052_arc0001_0239.jpg",
             "Sp_S_NRN_A_arc0059_arc0073_0240.jpg", "Sp_S_NRN_A_arc0069_arc0085_0241.jpg",
             "Sp_S_NRN_A_arc0080_arc0095_0243.jpg", "Sp_S_NRN_A_art0022_art0023_0244.jpg",
             "Sp_S_NRN_R_arc0011_arc0045_0085.jpg"]

base_dir = "/home/brechtl/Pictures/Data/CASIA/CASIA1/"

Path(base_dir + "original").mkdir(parents=True, exist_ok=True)
Path(base_dir + "forged").mkdir(parents=True, exist_ok=True)

for filename in to_remove:
    if os.path.exists(base_dir + "Sp/" + filename):
        os.remove(base_dir + "Sp/" + filename)

if os.path.exists(base_dir + "Au/"):
    for filename in os.listdir(base_dir + "Au/"):
        shutil.move(base_dir + "Au/" + filename, base_dir + "original/" + filename)
    shutil.rmtree(base_dir + "Au/")

if os.path.exists(base_dir + "Sp/"):
    for filename in os.listdir(base_dir + "Sp/"):
        if "_S_" in filename:
            shutil.move(base_dir + "Sp/" + filename, base_dir + "forged/" + filename)
    shutil.rmtree(base_dir + "Sp/")


# CASIAv2
# 7492 -> original
# 3295 -> forged
# 10787 -> total

base_dir = "/home/brechtl/Pictures/Data/CASIA/CASIA2/"

Path(base_dir + "original").mkdir(parents=True, exist_ok=True)
Path(base_dir + "forged").mkdir(parents=True, exist_ok=True)

if os.path.exists(base_dir + "Au/"):
    for filename in os.listdir(base_dir + "Au/"):
        shutil.move(base_dir + "Au/" + filename, base_dir + "original/" + filename)
    shutil.rmtree(base_dir + "Au/")

if os.path.exists(base_dir + "Tp/"):
    for filename in os.listdir(base_dir + "Tp/"):
        if "Tp_S_" in filename:
            shutil.move(base_dir + "Tp/" + filename, base_dir + "forged/" + filename)
    shutil.rmtree(base_dir + "Tp/")

# GRIP
# 0 -> original
# 80 -> forged
# 80 -> total

base_dir = "/home/brechtl/Pictures/Data/GRIP/"

Path(base_dir + "forged").mkdir(parents=True, exist_ok=True)

if os.path.exists(base_dir + "Forged Images/"):
    for filename in os.listdir(base_dir + "Forged Images/"):
        shutil.move(base_dir + "Forged Images/" + filename, base_dir + "forged/" + filename)
    shutil.rmtree(base_dir + "Forged Images/")
