import matplotlib.pyplot as plt
import numpy as np


def print_histogram_of_errors_with_cumsum(rel_loss_errs, sup_loss_errs):
    # https://matplotlib.org/gallery/ticks_and_spines/multiple_yaxis_with_spines.html

    frame_errs_rel = rel_loss_errs.mean(axis=1) / 10.
    frame_errs_sup = sup_loss_errs.mean(axis=1) / 10.

    fig, ax1 = plt.subplots()
    axis_font = {'size':'14'}
    ax1.set_ylabel("Num. Predicted 3D Poses", **axis_font)#,color='blue')
    ax1.set_xlabel("Reconstructed 3D Pose Error (cm)", **axis_font)
    # ax1.set_xlim([0,1000])
    # ax1.grid()
    # ax1.set_yscale('log', nonposy='clip')
    ax2 = ax1.twinx()
    ax2.set_ylim(0,1)
    ax2.set_xlim(0,20)
    ax2.set_yticks([.25,.5,.683,.954])
    ax2.set_yticklabels(['25th', '50th', '68th', '95th'])
    ax2.grid()
    ax2.set_axisbelow(True)
    ax1.set_axisbelow(True)
    ax2.set_ylabel("Percentiles", **axis_font)

    bin_size = 0.1

    ## rel loss
    print("Rel Loss (mean)  : ", round(np.mean(frame_errs_rel), 3))
    print("Rel Loss (median): ", round(np.median(frame_errs_rel), 3))
    c_rel = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
    num_bins = int(max(frame_errs_rel) / bin_size)
    n, bins, patches = ax1.hist(frame_errs_rel, num_bins,color=c_rel,alpha=.65, label=[r"$\bf{Ours}$" + " " +r"$\bf{Relative}$"])
    #print(n, np.sum(n))
    #print(bins)
    cum_sum = np.cumsum(n)/np.sum(n)
    #print(np.cumsum(n))
    #print(cum_sum)
    ax2.plot(bins[1:],cum_sum,color=c_rel)

    quarter_bin = np.argmin(abs((np.cumsum(n)/np.sum(n))-.25))
    median_bin = np.argmin(abs((np.cumsum(n)/np.sum(n))-.5))
    sigma_bin = np.argmin(abs((np.cumsum(n)/np.sum(n))-.683))
    two_sigma_bin = np.argmin(abs((np.cumsum(n)/np.sum(n))-.954))
    ax2.scatter([bins[quarter_bin+1],bins[median_bin+1],bins[sigma_bin+1],bins[two_sigma_bin+1]],
                [cum_sum[quarter_bin],cum_sum[median_bin],cum_sum[sigma_bin],cum_sum[two_sigma_bin]],c=c_rel)

    ax2.plot([bins[quarter_bin+1],bins[quarter_bin+1]],[0,cum_sum[quarter_bin]],'--',color=c_rel)
    ax2.plot([bins[median_bin+1],bins[median_bin+1]],[0,cum_sum[median_bin]],'--',color=c_rel)
    ax2.plot([bins[sigma_bin+1],bins[sigma_bin+1]],[0,cum_sum[sigma_bin]],'--',color=c_rel)
    ax2.plot([bins[two_sigma_bin+1],bins[two_sigma_bin+1]],[0,cum_sum[two_sigma_bin]],'--',color=c_rel)

    ## supervised
    print("Supervised (mean)  : ", round(np.mean(frame_errs_sup), 3))
    print("Supervised (median): ", round(np.median(frame_errs_sup), 3))
    c_sup = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    num_bins = int(max(frame_errs_sup) / bin_size)
    n, bins, patches = ax1.hist(frame_errs_sup, num_bins,color=c_sup,alpha=.65, label=[r"$\bf{3D}$" + " " +r"$\bf{Supervised}$"])
    #print(n, np.sum(n))
    #print(bins)
    cum_sum = np.cumsum(n)/np.sum(n)
    #print(np.cumsum(n))
    #print(cum_sum)
    ax2.plot(bins[1:],cum_sum,color=c_sup)

    quarter_bin = np.argmin(abs((np.cumsum(n)/np.sum(n))-.25))
    median_bin = np.argmin(abs((np.cumsum(n)/np.sum(n))-.5))
    sigma_bin = np.argmin(abs((np.cumsum(n)/np.sum(n))-.683))
    two_sigma_bin = np.argmin(abs((np.cumsum(n)/np.sum(n))-.954))
    ax2.scatter([bins[quarter_bin+1],bins[median_bin+1],bins[sigma_bin+1],bins[two_sigma_bin+1]],
                [cum_sum[quarter_bin],cum_sum[median_bin],cum_sum[sigma_bin],cum_sum[two_sigma_bin]],c=c_sup)

    ax2.plot([bins[quarter_bin+1],bins[quarter_bin+1]],[0,cum_sum[quarter_bin]],'--',color=c_sup)
    ax2.plot([bins[median_bin+1],bins[median_bin+1]],[0,cum_sum[median_bin]],'--',color=c_sup)
    ax2.plot([bins[sigma_bin+1],bins[sigma_bin+1]],[0,cum_sum[sigma_bin]],'--',color=c_sup)
    ax2.plot([bins[two_sigma_bin+1],bins[two_sigma_bin+1]],[0,cum_sum[two_sigma_bin]],'--',color=c_sup)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], loc='upper right')

    plt.show()
    # plt.savefig('plots/overall_errors.pdf', bbox_inches='tight')
    # plt.savefig('plots/overall_errors.png', bbox_inches='tight')
    # plt.close()

    print("Max  errors: ", frame_errs_sup.max(), frame_errs_rel.max())
    print("Min  errors: ", frame_errs_sup.min(), frame_errs_rel.min())
    print("Med  errors: ", np.median(frame_errs_sup), np.median(frame_errs_rel))


if __name__ == '__main__':
    rel_loss_errs = np.load('../checkpoint/fig_4ab/rel_supervised/all_dist_proc.npy')
    sup_loss_errs = np.load('../checkpoint/fig_4ab/fully_supervised/all_dist_proc.npy')
    print_histogram_of_errors_with_cumsum(rel_loss_errs, sup_loss_errs)
