import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
from typing import *
import pandas as pd

"""
# command to generate plots in the paper:
python scripts/plot/certified_acc_1seed.py --logpath "logs/XXXX/certification_log_50000.txt" --outdir . 
"""
# matplotlibrc params to set for better, bigger, clear plots
SMALLER_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20
plt.rc('font', size=BIGGER_SIZE)   # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


parser = argparse.ArgumentParser(description='Plot some plots')
# parser.add_argument('--logpath', type=str, help='path for the certified acc log txt file, e.g. xxx/certification_log_50000.txt')
parser.add_argument('--outdir', type=str, help='dir path for saving the fig')
args = parser.parse_args()

def at_radius(df: pd.DataFrame, radius: float):
    # breakpoint()
    return (df["correct"] & (df["radius"] >= radius)).mean()

x_label = r"$L_{2}$ radius"
max_radius=6.5
radius_step=0.05
figtext = "None"
radii = np.arange(0, max_radius + radius_step, radius_step)

df1 = pd.read_csv('logs/lambda1/scale_sig1-1/seed0/certify_diffpure', delimiter="\t")
print(df1["correct"].value_counts()[1])
line1 = np.array([at_radius(df1, radius) for radius in radii])
print(line1[0])

df2 = pd.read_csv('logs/lambda1/scale_sig1-5e-1/seed0/certify_diffpure', delimiter="\t")
print(df2["correct"].value_counts()[1])
line2 = np.array([at_radius(df2, radius) for radius in radii])
print(line2[0])

df3 = pd.read_csv('logs/lambda1/scale_sig1-1e-1/seed0/certify_diffpure', delimiter="\t")
print(df3["correct"].value_counts()[1])
line3 = np.array([at_radius(df3, radius) for radius in radii])
print(line3[0])

df4 = pd.read_csv('logs/lambda1/scale_sig1-1e-2/seed0/certify_diffpure', delimiter="\t")
print(df4["correct"].value_counts()[1])
line4 = np.array([at_radius(df4, radius) for radius in radii])
print(line4[0])

df11 = pd.read_csv('logs/imagenet-densepure-sample_num_10000-noise_1.41421356237-10-1_diffpure', delimiter="\t")
print(df11["correct"].value_counts()[1])
line11 = np.array([at_radius(df11, radius) for radius in radii])
print(line11[0])



# start the figure
plt.figure()
plt.plot(radii, line1, label="scale 1")
plt.plot(radii, line2, label="scale 3")
plt.plot(radii, line3, label="scale 5")
plt.plot(radii, line4, label="scale 7")
# plt.plot(radii, line5, label="scale 5")
# plt.plot(radii, line6, label="scale 6")
# plt.plot(radii, line7, label="scale 7")
# plt.plot(radii, line8, label="scale 10")
# plt.plot(radii, line9, label="scale 50")
# plt.plot(radii, line10, label="scale 100")
plt.plot(radii, line11, label="denp")


tick_frequency = 1
plt.ylim((0, 1))
plt.xlim((0, max_radius)) 
plt.tick_params()
plt.xlabel(x_label, labelpad=20, fontsize=BIGGER_SIZE)
plt.ylabel("Certified Accuracy", labelpad=20, fontsize=BIGGER_SIZE)
if figtext != "None":
    plt.figtext(0.05, 0.05, figtext)
plt.xticks(np.arange(0, max_radius+0.5, tick_frequency))
plt.legend(loc='upper right',fontsize="15")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir,"easy guide scaling 2.png"), dpi=300)
plt.close()
