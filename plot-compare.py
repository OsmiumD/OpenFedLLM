import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from argparse import ArgumentParser
import json

# parser = ArgumentParser()
# parser.add_argument('--filename', type=str, required=True)
# parser.add_argument('--dataset', type=bool, default=False)
# args = parser.parse_args()

filename1 = 'alp-Mat-Cod_20000_fedavg_c3s3_i10_b16a1_l1024_r32a64_20250120112122'
filename2 = 'alp-Mat-Cod_20000_fedavg_c3s3_i10_b16a1_l1024_r32a64_20250120131139'
plot_dataset = True
# 加载 training_loss.npy 文件
training_loss1 = np.load(os.path.join('./output', filename1, 'training_loss.npy'))
training_loss2 = np.load(os.path.join('./output', filename2, 'training_loss.npy'))
epochs = training_loss1.shape[1]

# load dataset client count
ds_skip_1 = 0
with open(os.path.join('./output', filename1, 'args.json')) as f_args:
    data = json.load(f_args)
    ds_skip_1 = data['fed_args']['num_clients_dataset']

ds_skip_2 = 0
with open(os.path.join('./output', filename2, 'args.json')) as f_args:
    data = json.load(f_args)
    ds_skip_2 = data['fed_args']['num_clients_dataset']

if not os.path.exists('./plots'):
    os.makedirs('./plots')

loss_lines1 = []
loss_lines2 = []
if plot_dataset:
    for i in range(training_loss1.shape[0] // ds_skip_1):
        loss_lines1.append(training_loss1[ds_skip_1 * i:ds_skip_1 * (i + 1)])
        loss_lines2.append(training_loss2[ds_skip_2 * i:ds_skip_2 * (i + 1)])

loss_lines1.append(training_loss1)
loss_lines2.append(training_loss2)

# 绘制每一行的损失曲线
def plot_avg_line(loss_group, idx, compare_id):
    training_loss_line = np.zeros(epochs)
    for i in range(epochs):
        line = loss_group[:, i]
        if len(line[line > -0.9]) > 0:
            training_loss_line[i] = np.average(line[line > -0.9])
        else:
            training_loss_line[i] = 0 if i == 0 else training_loss_line[i - 1]
    if idx == len(loss_lines1) - 1:
        plt.plot(training_loss_line, label=f'Average Loss - {compare_id}')
    else:
        plt.plot(training_loss_line, label=f'Train Loss {idx} - {compare_id}')
    print(f'Average loss on line{idx} test {compare_id}: {np.average(training_loss_line)}')

for idx in range(len(loss_lines1)):
    plt.figure()

    plot_avg_line(loss_lines1[idx], idx, 1)
    plot_avg_line(loss_lines2[idx], idx, 2)

    plt.title('Training Loss Over Time for Each Line')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'./plots/{filename1}_{filename2}{"_dataset" if plot_dataset else ""}_training_loss_{idx}.png')

# 保存图表为图片


# 如果不需要显示图表，可以注释掉以下行
# plt.show()

# num_clients_dataset
