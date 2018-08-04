import matplotlib.pyplot as plt
import numpy as np

def print_errors_over_kpt_type(rel_loss_errs, sup_loss_errs):

    fig, ax1 = plt.subplots()

    med_kpts = [0,7,8,9,10]
    right_kpts = [1,2,3, 14,15,16]
    left_kpts = [4,5,6,11,12,13]
    labels = ['mid_hip','mid_spine','neck','chin','head','hips','knees','ankles','shoulders','elbows','wrists']

    right_left_combined = np.vstack((rel_loss_errs[:,right_kpts],rel_loss_errs[:,left_kpts]))
    rel_vals = np.hstack(((rel_loss_errs[:,med_kpts]/10.).mean(axis=0), (right_left_combined/10.).mean(axis=0)))
    rel_stds = np.hstack(((rel_loss_errs[:,med_kpts]/10.).std(axis=0), (right_left_combined/10.).std(axis=0)))

    right_left_combined = np.vstack((sup_loss_errs[:,right_kpts],sup_loss_errs[:,left_kpts]))
    sup_vals = np.hstack(((sup_loss_errs[:,med_kpts]/10.).mean(axis=0), (right_left_combined/10.).mean(axis=0)))
    sup_stds = np.hstack(((sup_loss_errs[:,med_kpts]/10.).std(axis=0), (right_left_combined/10.).std(axis=0)))

    x = np.arange(len(labels))
    width = 0.45
    rects2 = ax1.bar(x, sup_vals, width, yerr=sup_stds, label=r"$\bf{3D}$" + " " +r"$\bf{Supervised}$")
    rects1 = ax1.bar(x + width, rel_vals, width, yerr=rel_stds, label=r"$\bf{Ours}$" + " " +r"$\bf{Relative}$")

    ax1.set_xticks(x)
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(labels,rotation=45)
    axis_font = {'size':'14'}
    ax1.set_ylabel("Average Keypoint Error (cm)", **axis_font)#,color='blue')
    ax1.set_xlabel("Keypoint Type", **axis_font)
    ax1.grid()
    ax1.set_axisbelow(True)

    plt.legend(loc='upper left')
    plt.show()
    # plt.savefig('plots/keypoint_error_bars.pdf', bbox_inches='tight')
    # plt.savefig('plots/keypoint_error_bars.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    rel_loss_errs = np.load('../checkpoint/fig_4ab/rel_supervised/all_dist_proc.npy')
    sup_loss_errs = np.load('../checkpoint/fig_4ab/fully_supervised/all_dist_proc.npy')
    print_errors_over_kpt_type(rel_loss_errs, sup_loss_errs)
