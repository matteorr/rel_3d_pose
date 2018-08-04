import numpy as np
import matplotlib.pyplot as plt
import json


def plot_figure(rel_noise_vs_accuracy, NOISE_AMOUNT_REL):
    fig, ax1 = plt.subplots()
    axis_font = {'size': '14'}
    ax1.set_xlabel("Percentage of Corrupted Relative Labels", **axis_font)
    ax1.set_ylabel("Reconstructed 3D Pose Error (cm)", **axis_font)
    ax1.grid()
    ax1.set_axisbelow(True)

    labels = NOISE_AMOUNT_REL
    for ii in range(len(labels)):
        if labels[ii] == 'mturk':
            labels[ii] = 'ANN NOISE'
        else:
            assert(type(labels[ii]) is not str)
            labels[ii] = str(int(labels[ii]*100)) + '%'
    x = np.arange(len(labels))

    width = 0.9
    ax1.bar(x, rel_noise_vs_accuracy, width, color='darkorange', label=r"$\bf{Ours}$" + " " +r"$\bf{Relative}$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    plt.legend()
    plt.show()
    # plt.savefig('plots/rel_noise_vs_accuracy.pdf', bbox_inches='tight')
    # plt.savefig('plots/rel_noise_vs_accuracy.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    res_file = '../checkpoint/fig_4d/fig_4d_results.json'
    with open(res_file) as da:
        data = json.load(da)
    rel_noise_vs_accuracy = [rr['err_test_last']/10.0 for rr in data]
    train_amount_noise = [rr['amt_noise'] for rr in data]

    plot_figure(rel_noise_vs_accuracy, train_amount_noise)
