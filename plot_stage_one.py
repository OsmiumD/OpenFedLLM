import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from argparse import ArgumentParser
import json

from matplotlib import ticker

# parser = ArgumentParser()
# parser.add_argument('--filename', type=str, required=True)
# parser.add_argument('--dataset', type=bool, default=False)
# args = parser.parse_args()

filename = 'alp-Mat-Cod_20000_fedavg_c3s3_i20_b16a1_l1024_r32a64_20250306134303'
plot_dataset = True
# 加载 training_loss.npy 文件
training_loss = np.load(os.path.join('./output', filename, 'stage_one_loss.npy'))

# load dataset client count
ds_skip = 0
with open(os.path.join('./output', filename, 'args.json')) as f_args:
    data = json.load(f_args)
    ds_skip = data['fed_args']['num_clients_dataset']

if not os.path.exists('./plots'):
    os.makedirs('./plots')

loss_lines = []
if plot_dataset:
    for i in range(training_loss.shape[0] // ds_skip):
        loss_lines.append(training_loss[ds_skip * i:ds_skip * (i + 1)])

loss_lines.append(training_loss)

# 绘制每一行的损失曲线
for idx, ds_loss in enumerate(loss_lines):
    training_loss_line = np.zeros(training_loss.shape[1])
    for i in range(training_loss.shape[1]):
        line = loss_lines[idx][:, i]
        if len(line[line > -0.9]) > 0:
            training_loss_line[i] = np.average(line[line > -0.9])
        else:
            training_loss_line[i] = 0 if i == 0 else training_loss_line[i - 1]
    if idx == len(loss_lines) - 1:
        plt.plot(training_loss_line, label=f'Average Loss')
    else:
        plt.plot(training_loss_line, label=f'Train Loss {idx}')

plt.title('Training Loss Over Time for Each Line')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.legend()
plt.grid()

# 保存图表为图片
plt.savefig(f'./plots/{filename}{"_dataset" if plot_dataset else ""}_stage_one_loss.png')

# 如果不需要显示图表，可以注释掉以下行
# plt.show()
