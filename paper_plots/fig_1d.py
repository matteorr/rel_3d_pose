import matplotlib.pyplot as plt
import json

def plot_figure(sup_accuracy, rel_accuracy, sup_data_amt, rel_data_amt,
                sup_train_data_perc, rel_train_data_perc, rel_num_pairs):

    fig, ax1 = plt.subplots()
    axis_font = {'size':'14'}
    ax1.set_xlabel("Amount of Training Data (MB)", **axis_font)
    ax1.set_ylabel("Reconstructed 3D Pose Error (cm)", **axis_font)
    ax1.grid()
    ax1.set_axisbelow(True)

    markers = ['+', 'd', 'p', 's', '^', 'o']
    sup_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    rel_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

    # set up labels
    sup_label = []
    for ii, pp in enumerate(sup_train_data_perc):
        perc = pp*100
        if (int(perc) - perc) == 0:
            perc = int(perc)
        sup_label.append(str(perc) + '% data')

    rel_label = []
    for ii, pp in enumerate(rel_train_data_perc):
        perc = pp*100
        if (int(perc) - perc) == 0:
            perc = int(perc)
        if rel_num_pairs[ii] == 1:
            rel_label.append(str(rel_num_pairs[ii]) + ' pair, ' + str(perc) + '% data')
        else:
            rel_label.append(str(rel_num_pairs[ii]) + ' pairs, ' + str(perc) + '% data')

    # plot
    for ii in range(len(sup_accuracy)):
        ax1.scatter(sup_data_amt[ii], sup_accuracy[ii], color=sup_color, label=sup_label[ii], marker=markers[ii], s=150)

    for ii in range(len(rel_accuracy)):
        ax1.scatter(rel_data_amt[ii], rel_accuracy[ii], color=rel_color, label=rel_label[ii], marker=markers[ii], s=150)

    ax1.set_xscale("log")
    #ax1.set_xlim(0.01,100)
    ax1.set_ylim(3.5,7.5)

    # create title in legend
    h, _ = ax1.get_legend_handles_labels()
    ph = [ax1.plot([],marker="", ls="")[0]]*2
    handles = [ph[0]] + h[:len(sup_accuracy)] + [ph[1]] + h[len(sup_accuracy):]
    labs = [r"$\bf{3D}$" + " " +r"$\bf{Supervised}$"] + sup_label + [r"$\bf{Ours}$" + " " +r"$\bf{Relative}$"] + rel_label

    ax1.legend(handles, labs, ncol=2)
    #ax1.legend(ncol=2)

    plt.show()
    # plt.savefig('plots/data_amount_vs_accuracy.pdf', bbox_inches='tight')
    # plt.savefig('plots/data_amount_vs_accuracy.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    with open('../checkpoint/fig_1d/fig_1d_results.json') as da:
        data = json.load(da)

    rel_accuracy = [dd/10.0 for dd in data['rel_supervised']]
    sup_accuracy = [dd/10.0 for dd in data['fully_supervised']]
    sup_train_data_perc = data['sup_train_data_perc']
    rel_train_data_perc = data['rel_train_data_perc']
    rel_num_pairs = data['rel_num_pairs']

    # compute size of train data in bits
    num_train_ims = 1559752  # size of train set
    bit_in_MB = 0.000000125  # bits in MegaBytes
    amt_train_single_bit = num_train_ims * bit_in_MB
    amt_train_sup = num_train_ims * 17 * 3 * 8 * bit_in_MB   # 17 keypoints, 3 joints, 8 bits for quantized depth

    sup_data_amt = [sup_train_data_perc[pp]*amt_train_sup for pp in range(len(sup_train_data_perc))]
    rel_data_amt = []
    for pp in range(len(rel_train_data_perc)):
        nps = rel_num_pairs[pp]
        if nps == 'all':
            nps = 136 # 17 choose 2, inpractice this will be upper bounded by the number of train epochs
        amt = rel_train_data_perc[pp] * amt_train_single_bit *nps
        rel_data_amt.append(amt)

    plot_figure(sup_accuracy, rel_accuracy, sup_data_amt, rel_data_amt,
                sup_train_data_perc, rel_train_data_perc, rel_num_pairs)
