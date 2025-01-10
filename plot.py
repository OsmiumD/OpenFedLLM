import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 加载 training_loss.npy 文件
training_loss = np.load(os.path.join('./output', sys.argv[1], 'training_loss.npy'))

print(training_loss)
# 绘制每一行的损失曲线
training_loss_line = np.zeros(training_loss.shape[1])
for i in range(training_loss.shape[1]):
    line = training_loss[:, i]
    training_loss_line[i] = np.average(line[line > -0.9])
# all_idx = np.array(list(range(training_loss.shape[1])))
# for i in [0, 5, 10, 15]:
#     line = training_loss[i]
plt.plot(training_loss_line, label=f'Train Loss')

plt.title('Training Loss Over Time for Each Line')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# 保存图表为图片
plt.savefig(f'./plots/training_loss_lines_plot_{sys.argv[1]}.png')

# 如果不需要显示图表，可以注释掉以下行
# plt.show()
