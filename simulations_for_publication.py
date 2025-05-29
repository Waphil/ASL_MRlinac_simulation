import numpy as np
from util import t1a_estimate, t2a_estimate, calculate_fwhm
from visualize import basic_multiline_plot
from fair_bssfp_simulation import simulate_bssfp_psf, simulate_fair_bssfp_signal_difference_psf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Creating figures for the publication
##################################################################


def main():
    # Plot parameters
    figsize = (4.416, 3.4)
    labels_fontsize = 11 #None #12

    # Hardware / constants
    b0       = 0.35                   # [T]
    bw_px_min = 130                   # [Hz/px] (minimum bandwidth per pixel)
    t_grad   = 3.43 #2.7              # [ms]    (dead time within TR for gradients/RF pulses)
    tr_max_bw  = 1e3/bw_px_min+t_grad # [ms]    (maximum TR to assure high enough BW)
    delta_f_max = 40                  # [Hz]    (maximum off-resonance in the brain)
    tr_max_offres = 1e3/(3*delta_f_max) # [ms]    (maximum TR avoiding artefacts, Bieri et al.)
    tr_max    = min(tr_max_bw, tr_max_offres)

    # Image settings constants
    n        = 96                   #         (number of pixels in x and y direction)

    # Simulation constants
    m0       = 1                    # [A/m]
    t1a      = t1a_estimate(b0, 2)   # [ms]
    t2a      = t2a_estimate(b0, 1)   # [ms]
    delta_t  = 500                  # [ms]    (transit time) # TODO: check why changed to 500
    tau      = 1800                 # [ms]    (trailing time)
    partition_coef = 0.9            # [ml/g]  (blood-tissue partition coefficient)
    inv_eff  = 0.8                  #         (inversion efficiency) # TODO: check why / if this value is chosen!!! # Maybe go for 0.98 instead
    q        = 1                    #         (correction factor (normally close to unity))
    perf     = 1                    #         (perfusion)

    # Image settings that are more specific
    n_dummy_tr  = 8                 #         (number of dummy rf pulses every bSSFP readout before signal is written)
    d_dummy_tr2 = 0                 #         (number of dummy inversion pulses before readout start)

    ###########################
    # Experiment: Variable TR #
    ###########################

    fa       = 110 # TODO: check if this is really a good idea
    n_dummy_tr  = 8
    tr_arr   = np.linspace(5., 8., num=4, endpoint=True).reshape((1, -1))
    ti       = 900

    # Calculating PSFs
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa, tr_arr, ti, t1a, t2a, inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n)
    z = z[:, 0]
    psf = psf.T

    # Plot settings
    grid_kwargs = dict(which="major", axis="both")
    x_label = "pixel coordinate"
    y_label = "PSF [A.U.]"
    x_lim = [-3.5, 3.5]
    y_lim = [-10., 100.]
    label_list = [f"{tr}" for tr in tr_arr.ravel()]
    legend_title = "Repetition\nTime [ms]"
    legend_kwargs = dict(loc='upper right', fancybox=True, shadow=False, framealpha=1.)
    colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F"]
    basic_multiline_plot(z, psf, label_list, # TODO: investigate if should show FWHM
                         ax=None, figsize=figsize, colors=colors, linestyles=None, alphas=None,
                         title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                         legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)

    ###########################
    # Experiment: Variable FA #
    ###########################

    fa_arr = np.linspace(70, 130, num=4, endpoint=True).reshape((1, -1))
    n_dummy_tr_arr = 8. #np.around(fa_arr*0.05) # TODO: check this heuristic formula
    tr       = 6.6
    ti       = 900

    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr, ti, t1a, t2a, inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr_arr, n_tr=n)
    z = z[:, 0]
    psf = psf.T


    grid_kwargs = dict(which="major", axis="both")
    x_label = "pixel coordinate"
    y_label = "PSF [A.U.]"
    x_lim = [-4, 4]
    y_lim = [-10., 100.]
    label_list = [f"{fa}" for fa in fa_arr.ravel()]
    legend_title = "Excitation\nFlip Angle [°]"
    legend_kwargs = dict(loc='upper right', fancybox=True, shadow=False, framealpha=1.)
    colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F"]
    basic_multiline_plot(z, psf, label_list,
                         ax=None, figsize=figsize, colors=colors, linestyles=None, alphas=None,
                         title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                         legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=True)


    #######################################
    # Experiment: PSF SNR for variable TR #
    #######################################

    # Parameters
    fa   = 90 # TODO: Why different from above?
    tr_arr = np.linspace(4., 9.2, num=261, endpoint=True).reshape((1, -1, 1)) # 4:0.01:9.2
    ti   = 1300 # TODO: Why different from above?

    # Relaxation Parameter Variability
    vary_factor   = 0.1
    t1a_arr       = (1+vary_factor*np.array([-1, 0, 1]))*t1a
    t2a_arr       = (1+vary_factor*np.array([-1, 0, 1]))*t2a
    #[T1var,T2var] = meshgrid(T1s,T2s)

    total_t1a_arr, total_t2a_arr = np.meshgrid(t1a_arr, t2a_arr)
    total_t1a_arr = np.reshape(total_t1a_arr, (1, 1, -1))
    total_t2a_arr = np.reshape(total_t2a_arr, (1, 1, -1))

    # Calculating PSFs
    # few points to make it computationally manageable. We only care about center here anyway.
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa, tr_arr, ti, total_t1a_arr, total_t2a_arr,
                                                       inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=10)
    z = z[:, 0]
    psf_max_arr = np.amax(psf, axis=0) # Center of PSF is the max value in each case

    snr_arr = psf_max_arr * np.sqrt(tr_arr.reshape((-1, 1))-t_grad) # tr_arr - t_grad should be roughly inversely proportional to pixelBW

    # Normalize SNR
    snr_norm_arr = snr_arr/np.amax(snr_arr, axis=0) # TODO: think about normalization implications

    snr_norm_for_plot_arr = snr_norm_arr.T

    # Find the lowest value that gets SNR to 95%
    first_tr_with_high_snr = tr_arr.ravel()[np.nonzero(snr_norm_for_plot_arr[4] > 0.95)[0][0]]

    # Plot

    grid_kwargs = dict(which="major", axis="both")
    x_label = "TR [ms]"
    y_label = r"$SNR_{rel}$"
    x_lim = [np.amin(tr_arr), np.amax(tr_arr)]
    y_lim = None
    label_list = ["" for i in snr_norm_for_plot_arr]
    legend_title = "Scenario"
    legend_kwargs = dict(loc='lower right', fancybox=True, shadow=False, framealpha=1.)
    colors = [None, "#D95319", None, "#77AC30", "#0072BD", "#77AC30", None, "#D95319", None]
    linestyles = ["", ":", "", ":", "-", ":", "", ":", ""]
    ax = basic_multiline_plot(tr_arr.ravel(), snr_norm_for_plot_arr, label_list,
                              ax=None, figsize=figsize, colors=colors, linestyles=linestyles, alphas=None,
                              title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                              is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                              x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                              x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                              legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=1)
    ax.axvline(first_tr_with_high_snr, color="gray", linestyle=":", linewidth=1)

    # Define custom legend for aesthetic reasons
    legend_handles = [
        Line2D([0], [0], color="#0072BD", marker=None, linestyle="-", label="baseline"),
        Line2D([0], [0], color="#77AC30", marker=None, linestyle="--", label=r"vary $T_1$"),
        Line2D([0], [0], color="#D95319", marker=None, linestyle="--", label=r"vary $T_2$"),
    ]
    ax.legend(handles=legend_handles, framealpha=1.)

    plt.show()

    #######################################
    # Experiment: PSF SNR for variable FA #
    #######################################

    # Parameters
    fa_arr   = np.linspace(45., 155., num=111, endpoint=True).reshape((1, -1, 1)) # 45:1:155
    tr = 6. # TODO: Why different from above?
    ti   = 1300 # TODO: Why different from above?
    t_grad = 2.7 # TODO: Why different again?

    # Relaxation Parameter Variability
    vary_factor   = 0.1
    t1a_arr       = (1+vary_factor*np.array([-1, 0, 1]))*t1a
    t2a_arr       = (1+vary_factor*np.array([-1, 0, 1]))*t2a
    #[T1var,T2var] = meshgrid(T1s,T2s)

    total_t1a_arr, total_t2a_arr = np.meshgrid(t1a_arr, t2a_arr)
    total_t1a_arr = np.reshape(total_t1a_arr, (1, 1, -1))
    total_t2a_arr = np.reshape(total_t2a_arr, (1, 1, -1))

    # Calculating PSFs
    # few points to make it computationally manageable. We only care about center here anyway.
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr, ti, total_t1a_arr, total_t2a_arr,
                                                       inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=10)
    z = z[:, 0]
    psf_max_arr = np.amax(psf, axis=0) # Center of PSF is the max value in each case

    snr_arr = psf_max_arr * np.sqrt(tr-t_grad) # tr - t_grad should be roughly inversely proportional to pixelBW

    # Normalize SNR
    snr_norm_arr = snr_arr/np.amax(snr_arr, axis=0) # TODO: think about normalization implications

    snr_norm_for_plot_arr = snr_norm_arr.T

    # Find the lowest value that gets SNR to 95%
    first_fa_with_high_snr = fa_arr.ravel()[np.nonzero(snr_norm_for_plot_arr[4] > 0.95)[0][0]]

    # Plot

    grid_kwargs = dict(which="major", axis="both")
    x_label = "FA [°]"
    y_label = r"$SNR_{rel}$"
    x_lim = [np.amin(fa_arr), np.amax(fa_arr)]
    y_lim = None
    label_list = ["" for i in snr_norm_for_plot_arr]
    legend_title = "Scenario"
    legend_kwargs = dict(loc='lower right', fancybox=True, shadow=False, framealpha=1.)
    colors = [None, "#D95319", None, "#77AC30", "#0072BD", "#77AC30", None, "#D95319", None]
    linestyles = ["", ":", "", ":", "-", ":", "", ":", ""]
    ax = basic_multiline_plot(fa_arr.ravel(), snr_norm_for_plot_arr, label_list,
                              ax=None, figsize=figsize, colors=colors, linestyles=linestyles, alphas=None,
                              title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                              is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                              x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                              x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                              legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=1)
    ax.axvline(first_fa_with_high_snr, color="gray", linestyle=":", linewidth=1)

    # Define custom legend for aesthetic reasons
    legend_handles = [
        Line2D([0], [0], color="#0072BD", marker=None, linestyle="-", label="baseline"),
        Line2D([0], [0], color="#77AC30", marker=None, linestyle="--", label=r"vary $T_1$"),
        Line2D([0], [0], color="#D95319", marker=None, linestyle="--", label=r"vary $T_2$"),
    ]
    ax.legend(handles=legend_handles, framealpha=1.)

    plt.show()


if __name__ == "__main__":
    main()