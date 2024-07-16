import os
import matplotlib.pyplot as plt


def save_fig(name):
    output_dir = '/Users/kerembuekrue/Documents/code/DeepImageception/output/'
    if len(name) > 1:
        output_dir += name[0]
        filename = name[1]
    else:
        filename = name[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename + ".png"), dpi=100)
