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

# mrqa-tinyllama
filename1 = 'mrq-mrq-mrq-mrq-mrq-mrq_10000_fedavg_c6s3_i20_b8a2_l2048_r32a64_20250324184648'  #compare
filename2 = 'mrq-mrq-mrq-mrq-mrq-mrq_10000_fedavg_c6s3_i20_b8a2_l2048_r32a64_20250324115623'  #no freeze


#gsm8k-tinyllama
# filename1 = 'gsm_500_fedavg_c6s3_i20_b16a1_l1024_r32a64_20250312081709'  #compare
# filename2 = 'gsm_500_fedavg_c6s3_i20_b16a1_l1024_r32a64_20250312063630'  #no freeze

# llama2-7b
# filename1 = 'alp-Mat-Cod_20000_fedavg_c3s3_i20_b16a1_l1024_r32a64_20250306143857'  # compare
# filename2 = 'alp-Mat-Cod_20000_fedavg_c3s3_i20_b16a1_l1024_r32a64_20250306224711'  # one_stage
# filename2 = 'alp-Mat-Cod_20000_fedavg_c3s3_i20_b16a1_l1024_r32a64_20250307075907'  # no freeze
# filename2 = 'alp-Mat-Cod_20000_fedavg_c3s3_i20_b16a1_l1024_r32a64_20250310124745'  # freeze



# filename1 = 'alp-Mat-Cod_20000_fedavg_c3s3_i20_b16a1_l1024_r32a64_20250228182602' # compare
# filename2 = 'alp-Mat-Cod_20000_fedavg_c3s3_i20_b16a1_l1024_r32a64_20250228152116' # use prompt

# filename1 = 'alp-Mat-Cod_20000_fedyogi_c3s3_i10_b16a1_l1024_r32a64_20250219055230'  # Same lr
# filename2 = 'alp-Mat-Cod_20000_fedyogi_c3s3_i10_b16a1_l1024_r32a64_20250227124303'  # New method

# filename1 = 'alp-Mat-Cod_20000_fedyogi_c3s3_i10_b16a1_l1024_r32a64_20250217202543'   # One Stage
# filename1 = 'alp-Mat-Cod_20000_fedyogi_c3s3_i10_b16a1_l1024_r32a64_20250217170650'   # Freeze Prompt
# filename1 = 'alp-Mat-Cod_20000_fedyogi_c3s3_i10_b16a1_l1024_r32a64_20250212135150'  # No Freeze
# filename2 = 'alp-Mat-Cod_20000_fedyogi_c3s3_i10_b16a1_l1024_r32a64_20250213070040'
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
