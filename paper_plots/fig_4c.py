import numpy as np
import matplotlib.pyplot as plt
import json


def plot_figure(rel_acc_te, sup_acc_te, rel_acc_tr_te, sup_acc_tr_te, labels, inds_of_interest):

    fig, ax1 = plt.subplots()
    markers = ['+', 'd', 'p', 's', '^', 'o']
    plt.plot([0,20], [0,20], ':', color='gray')

    sup_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    rel_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

    for ii in inds_of_interest:
        ax1.scatter(sup_acc_te[ii], sup_acc_tr_te[ii], color=sup_color, label=labels[ii], marker=markers[ii], s=150)

    for ii in inds_of_interest:
        ax1.scatter(rel_acc_te[ii], rel_acc_tr_te[ii], color=rel_color, label=labels[ii], marker=markers[ii], s=150)

    # create title in legend
    h, _ = ax1.get_legend_handles_labels()
    ph = [ax1.plot([],marker="", ls="")[0]]*2
    handles = [ph[0]] + h[:len(inds_of_interest)] + [ph[1]] + h[len(inds_of_interest):]
    labs = [r"$\bf{3D}$" + " " +r"$\bf{Supervised}$"] + [labels[ii] for ii in inds_of_interest] + [r"$\bf{Ours}$" + " " +r"$\bf{Relative}$"] + [labels[ii] for ii in inds_of_interest]

    axis_font = {'size':'14'}
    ax1.set_xlabel("3D Pose Error (cm) - Test Noise (GT/SH)", **axis_font)
    axis_font = {'size':'13'}
    ax1.set_ylabel("3D Pose Error (cm) - Train Test Noise (SH/SH)", **axis_font)
    ax1.yaxis.set_label_coords(-0.05,0.45)
    ax1.grid()
    ax1.set_axisbelow(True)

    ax1.legend(handles, labs, ncol=2)
    plt.xlim(3, 8)
    plt.ylim(3, 8)

    plt.show()
    # plt.savefig('plots/noise_2d_vs_accuracy_both.pdf', bbox_inches='tight')
    # plt.savefig('plots/noise_2d_vs_accuracy_both.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    res_file_test = '../checkpoint/fig_4c_test_noise/fig_4c_test_noise_results.json'
    with open(res_file_test) as da:
        data_test = json.load(da)

    res_file_train_test = '../checkpoint/fig_4c_train_test_noise/fig_4c_train_test_noise_results.json'
    with open(res_file_train_test) as da:
        data_train_test = json.load(da)

    rel_acc_te = np.array(data_test['rel_supervised'])/10.0
    sup_acc_te = np.array(data_test['fully_supervised'])/10.0
    amount_noise_te = data_test['test_amount_noise']
    amount_noise_te = ['DTS' if str(tt) == 'detections' else tt for tt in amount_noise_te]

    rel_acc_tr_te = np.array(data_train_test['rel_supervised'])/10.0
    sup_acc_tr_te = np.array(data_train_test['fully_supervised'])/10.0
    amount_noise_tr_te = data_train_test['test_amount_noise']
    amount_noise_tr_te = ['DTS' if str(tt) == 'detections' else tt for tt in amount_noise_tr_te]

    assert amount_noise_te == amount_noise_tr_te

    labels = []
    for aa in amount_noise_te:
        if aa == 'DTS':
            labels.append('DTS')
        else:
            #labels.append(str(aa) + '%')
            labels.append(r"$\sigma^2 = $" + str(aa))


    #inds_of_interest = [0, 1, 2, 3, 4, 5]  # use this only plot a subset
    inds_of_interest = [0, 2, 4, 5]  # use this only plot a subset

    plot_figure(rel_acc_te, sup_acc_te, rel_acc_tr_te, sup_acc_tr_te, labels, inds_of_interest)
