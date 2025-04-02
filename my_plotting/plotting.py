# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import json
import math
from typing import Any, Dict, List


import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.sans-serif': 'SimHei',
    'axes.unicode_minus': False,
    'axes.labelsize': 16,        # 坐标轴标签字体大小
    'axes.titlesize': 16,        # 图标题字体大小
    'legend.fontsize': 14,       # 图例字体大小
    'xtick.labelsize': 14,       # X轴刻度标签字体大小
    'ytick.labelsize': 14,       # Y轴刻度标签字体大小
    'axes.linewidth': 1.5,       # 坐标轴线宽
    'legend.frameon': False,     #设置图例不带框
    'xtick.minor.visible': True, # 开启坐标轴次要刻度线
    'ytick.minor.visible': True, # 开启坐标轴次要刻度线
    'xtick.top': True,           # 开启坐标轴上轴
    'ytick.right': True,         # 开启坐标轴右轴
    'savefig.dpi': 300,          # 保存图片DPI
    'figure.dpi': 300            # 绘图DPI
})



def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed



def plot_loss() -> None:
    r"""
    Plots loss curves and saves the image.
    """
    plt.switch_backend("agg")
    with open('my_plotting/trainer_state.json', encoding="utf-8") as f:
        data = json.load(f)

    step_data = {}
    for entry in data["log_history"]:
        step = entry['step']
        if step not in step_data:
            step_data[step] = {'train_loss': [], 'eval_loss': []}
        
        if 'loss' in entry:
            step_data[step]['train_loss'].append(entry['loss'])
        if 'eval_loss' in entry:
            step_data[step]['eval_loss'].append(entry['eval_loss'])

    steps = sorted(step_data.keys())
    train_losses = [d['train_loss'][0] if d['train_loss'] else None for d in step_data.values()]
    eval_losses = [d['eval_loss'][0] if d['eval_loss'] else None for d in step_data.values()]

    fig, ax =  plt.subplots()
    ax.plot(steps, train_losses, color='k')
    # plt.title("训练损失曲线图")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("loss")
    plt.xticks([])
    plt.yticks([])
    ax.set_xticks(np.arange(0, 351, 50))
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    # ax.grid(True)


    figure_path =f"my_plotting/training_loss.png"
    plt.savefig(figure_path, format="png", dpi=100)
    print("Figure saved at:", figure_path)

if __name__ == '__main__':
    plot_loss()