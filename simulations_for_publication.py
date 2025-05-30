import numpy as np
from util import t1a_estimate, t2a_estimate, calculate_fwhm
from visualize import basic_multiline_plot
from fair_bssfp_simulation import (simulate_bssfp_psf, simulate_fair_bssfp_signal_difference_psf,
                                   correction_factor_for_finite_tm)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Creating figures for the publication
##################################################################


def main():
    # Plot parameters
    figsize = (4.416, 3.4)
    labels_fontsize = 11 #None #12
    fwhm_line_height = 30.

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
    t_total  = 6*60*1000.           # [ms]    (total acquisition time constraint set for whole sequence in optimization)

    # Simulation constants
    m0       = 1                    # [A/m]
    t1a      = t1a_estimate(b0, 2)   # [ms]
    t2a      = t2a_estimate(b0, 1)   # [ms]
    delta_t  = 500.                 # [ms]    (transit time)
    tau      = 1800.                # [ms]    (trailing time)
    partition_coef = 0.9            # [ml/g]  (blood-tissue partition coefficient)
    inv_eff  = 0.98                 #         (inversion efficiency) # 0.98 recommended for pASL
    q        = 1                    #         (correction factor (normally close to unity))
    perf     = 1                    #         (perfusion)

    # Image settings that are more specific
    n_dummy_tr  = 8                 #         (number of dummy rf pulses every bSSFP readout before signal is written)
    n_dummy_tm  = 0                 #         (number of dummy inversion pulses before readout start)

    ##################################################
    # Optimize SNR across a wide range of parameters #
    ##################################################

    # Parameters
    fa_arr      = np.linspace(80., 130., num=41, endpoint=True).reshape((1, -1, 1, 1, 1)) #90:5:130     # [°]
    tr_arr      = np.linspace(5.6, 8.3, num=55, endpoint=True).reshape((1, 1, -1, 1, 1)) #5:0.2:7      # [ms]
    ti_arr      = np.linspace(1100., 1500., num=81, endpoint=True).reshape((1, 1, 1, -1, 1)) #1100:20:1400 # [ms]
    td_arr      = np.linspace(0., 400., num=21, endpoint=True).reshape((1, 1, 1, 1, -1)) #0:100:400    # [ms]
    #n_avg_arr   = np.linspace(60., 110., num=6, endpoint=True).reshape((1, 1, 1, 1, 1, -1)) #60:10:110

    # Total measurement time is time for inversion + rf pulses (including dummy) + dead time
    tm_arr = ti_arr + (n + n_dummy_tr) * tr_arr + td_arr

    # Calculate maximum number of averages (each average is 1 tag & 1 control image) without exceeding total acq. time
    n_avg_arr = np.floor(t_total / (2*tm_arr) - n_dummy_tm)

    # Calculate the array corresponding to readout time (tr - t_grad), which is the inverse of the pixelBW
    t_readout_arr = (tr_arr-t_grad)

    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr_arr, ti_arr, t1a, t2a, inv_eff, perf,
                                                       partition_coef, delta_t, tau, q_func=None,
                                                       n_dummy_tr=n_dummy_tr, n_tr=n, n_points_psf=1)
    z = z.reshape((-1,))
    psf_max = np.amax(psf, axis=0)

    snr_arr = psf_max * np.sqrt(n_avg_arr[0] * t_readout_arr[0]) # Have to remove first dimension of avg and readout to match them
    snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm_arr[0], t1a, inv_eff)

    #best_settings_index = np.argmax(snr_finitetm_corr_arr) # hmm, doesn't work
    best_settings_indices_masks = [
        np.count_nonzero(snr_finitetm_corr_arr == np.amax(snr_finitetm_corr_arr),
                         axis=tuple([j for j in range(snr_finitetm_corr_arr.ndim) if j != i]))
        for i in range(snr_finitetm_corr_arr.ndim)]
    best_settings_indices = [np.argmax(best_settings) for best_settings in best_settings_indices_masks]

    print(f"-----------------------\n"
          f"Optimized settings:\n"
          f"FA = {fa_arr.ravel()[best_settings_indices[0]]}°\n"
          f"TR = {tr_arr.ravel()[best_settings_indices[1]]}ms\n"
          f"TI = {ti_arr.ravel()[best_settings_indices[2]]}ms\n"
          f"TD = {td_arr.ravel()[best_settings_indices[3]]}ms\n"
          f"TM = {tm_arr[0,0][*best_settings_indices[1:]]}ms\n"
          f"N_avg = {n_avg_arr[0,0][*best_settings_indices[1:]]}\n")

    ###########################
    # Experiment: Variable TR #
    ###########################

    fa       = 95. #110 # TODO: check if this is really a good idea
    n_dummy_tr  = 8
    tr_arr   = np.linspace(5., 8., num=4, endpoint=True).reshape((1, -1))
    ti       = 1345. #900

    # Calculating PSFs
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa, tr_arr, ti, t1a, t2a, inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n)
    z = z[:, 0]
    psf = psf.T

    fwhm_tuple_list = [calculate_fwhm(z, psf_individual) for psf_individual in psf]

    # Plot settings
    grid_kwargs = dict(which="major", axis="both")
    x_label = r"pixel coordinate $z$"
    y_label = r"$PSF$ $\left[A.U.\right]$"
    x_lim = [-3.5, 3.5]
    y_lim = [-10., 150.]
    label_list = [f"{tr}" for tr in tr_arr.ravel()]
    legend_title = f"$\\alpha$ = {fa}°,\n$TR$ $\\left[ms\\right]$"
    legend_kwargs = dict(loc='upper right', fancybox=True, shadow=False, framealpha=1.)
    colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F"]
    ax = basic_multiline_plot(z, psf, label_list, # TODO: investigate if should show FWHM
                         ax=None, figsize=figsize, colors=colors, linestyles=None, alphas=None,
                         title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                         legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)
    for psf_individual, fwhm_tuple, color in zip(psf, fwhm_tuple_list, colors):
        x_coords = np.array([z[fwhm_tuple[1]], z[fwhm_tuple[2]]])
        y_coords = np.array([psf_individual[fwhm_tuple[1]], psf_individual[fwhm_tuple[2]]])
        ax.errorbar(x_coords, y_coords, yerr=fwhm_line_height, ecolor=color, fmt="", linestyle="")
    #plt.show()

    ###########################
    # Experiment: Variable FA #
    ###########################

    fa_arr = np.linspace(70, 130, num=4, endpoint=True).reshape((1, -1))
    n_dummy_tr_arr = 8. #np.around(fa_arr*0.05) # TODO: check this heuristic formula
    tr       = 7.65 #6.6
    ti       = 1345 #900

    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr, ti, t1a, t2a, inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr_arr, n_tr=n)
    z = z[:, 0]
    psf = psf.T

    fwhm_tuple_list = [calculate_fwhm(z, psf_individual) for psf_individual in psf]

    grid_kwargs = dict(which="major", axis="both")
    x_label = r"pixel coordinate $z$"
    y_label = r"$PSF$ $\left[A.U.\right]$"
    x_lim = [-3.5, 3.5]
    y_lim = [-10., 115.]
    label_list = [f"{fa}" for fa in fa_arr.ravel()]
    legend_title = f"$TR$ = {tr} ms,\n$\\alpha$ $\\left[°\\right]$"
    legend_kwargs = dict(loc='upper right', fancybox=True, shadow=False, framealpha=1.)
    colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F"]
    ax = basic_multiline_plot(z, psf, label_list,
                         ax=None, figsize=figsize, colors=colors, linestyles=None, alphas=None,
                         title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                         legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)
    for psf_individual, fwhm_tuple, color in zip(psf, fwhm_tuple_list, colors):
        x_coords = np.array([z[fwhm_tuple[1]], z[fwhm_tuple[2]]])
        y_coords = np.array([psf_individual[fwhm_tuple[1]], psf_individual[fwhm_tuple[2]]])
        ax.errorbar(x_coords, y_coords, yerr=fwhm_line_height, ecolor=color, fmt="", linestyle="")
    plt.show()

    #######################################
    # Experiment: PSF SNR for variable TR #
    #######################################

    # Parameters
    fa   = 95. #90
    tr_arr = np.linspace(4.5, 9.2, num=236, endpoint=True).reshape((1, -1, 1)) # 4:0.01:9.2
    ti   = 1345. #1300
    td   = 0.
    t_total = 100.*6*60*1000. # Do this to avoid discrete N_average phenomenons on the curve's appearance.

    # Calculate the array corresponding to readout time (tr - t_grad), which is the inverse of the pixelBW
    t_readout_arr = (tr_arr-t_grad)

    # Relaxation Parameter Variability
    vary_factor   = 0.1
    t1a_arr       = (1+vary_factor*np.array([-1, 0, 1]))*t1a
    t2a_arr       = (1+vary_factor*np.array([-1, 0, 1]))*t2a

    total_t1a_arr, total_t2a_arr = np.meshgrid(t1a_arr, t2a_arr)
    total_t1a_arr = np.reshape(total_t1a_arr, (1, 1, -1))
    total_t2a_arr = np.reshape(total_t2a_arr, (1, 1, -1))

    # Total measurement time is time for inversion + rf pulses (including dummy) + dead time
    tm_arr = ti + (n + n_dummy_tr) * tr_arr + td

    # Calculate maximum number of averages (each average is 1 tag & 1 control image) without exceeding total acq. time
    n_avg_arr = np.floor(t_total / (2*tm_arr) - n_dummy_tm)

    # Calculating PSFs
    # few points to make it computationally manageable. We only care about center here anyway.
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa, tr_arr, ti, total_t1a_arr, total_t2a_arr,
                                                       inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=1)
    z = z[:, 0]
    psf_max_arr = np.amax(psf, axis=0) # Center of PSF is the max value in each case

    snr_arr = psf_max_arr * np.sqrt(n_avg_arr.reshape((-1, 1)) * t_readout_arr.reshape((-1, 1)))

    snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm_arr[0], t1a, inv_eff)

    # Normalize SNR
    snr_norm_arr = snr_arr/np.amax(snr_arr, axis=0)
    snr_finitetm_corr_norm_arr = snr_finitetm_corr_arr/np.amax(snr_finitetm_corr_arr, axis=0)

    snr_norm_for_plot_arr = snr_norm_arr.T
    snr_finitetm_corr_arr_norm_for_plot_arr = snr_finitetm_corr_norm_arr.T

    # Find the lowest value that gets SNR to 95%
    first_tr_with_high_snr = tr_arr.ravel()[np.nonzero(snr_norm_for_plot_arr[4] > 0.95)[0][0]]

    # Plot

    grid_kwargs = dict(which="major", axis="both")
    x_label = r"$TR$ $\left[ms\right]$"
    y_label = r"$SNR_{rel}$"
    x_lim = [np.amin(tr_arr), np.amax(tr_arr)]
    y_lim = None
    label_list = ["" for i in snr_norm_for_plot_arr]
    legend_title = "Scenario"
    legend_kwargs = dict(loc='lower right', fancybox=True, shadow=False, framealpha=1.)
    colors = [None, "#D95319", None, "#77AC30", "#0072BD", "#77AC30", None, "#D95319", None]
    linestyles = ["", ":", "", ":", "-", ":", "", ":", ""]
    ax = basic_multiline_plot(tr_arr.ravel(), snr_finitetm_corr_arr_norm_for_plot_arr, label_list, #snr_norm_for_plot_arr
                              ax=None, figsize=figsize, colors=colors, linestyles=linestyles, alphas=None,
                              title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                              is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                              x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                              x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                              legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)
    #ax.axhline(0.95, color="gray", linestyle=":", linewidth=1)
    #ax.axvline(first_tr_with_high_snr, color="gray", linestyle=":", linewidth=1)

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
    tr = 7.65 #6.
    ti   = 1345. #1300
    t_grad = 3.43 #2.7

    # Calculate the array corresponding to readout time (tr - t_grad), which is the inverse of the pixelBW
    t_readout = (tr-t_grad)

    # Relaxation Parameter Variability
    vary_factor   = 0.1
    t1a_arr       = (1+vary_factor*np.array([-1, 0, 1]))*t1a
    t2a_arr       = (1+vary_factor*np.array([-1, 0, 1]))*t2a

    total_t1a_arr, total_t2a_arr = np.meshgrid(t1a_arr, t2a_arr)
    total_t1a_arr = np.reshape(total_t1a_arr, (1, 1, -1))
    total_t2a_arr = np.reshape(total_t2a_arr, (1, 1, -1))

    # Total measurement time is time for inversion + rf pulses (including dummy) + dead time
    tm = ti + (n + n_dummy_tr) * tr + td

    # Calculate maximum number of averages (each average is 1 tag & 1 control image) without exceeding total acq. time
    n_avg = np.floor(t_total / (2*tm) - n_dummy_tm)

    # Calculating PSFs
    # few points to make it computationally manageable. We only care about center here anyway.
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr, ti, total_t1a_arr, total_t2a_arr,
                                                       inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=1)
    z = z[:, 0]
    psf_max_arr = np.amax(psf, axis=0) # Center of PSF is the max value in each case

    snr_arr = psf_max_arr * np.sqrt(n_avg * t_readout) # Meaningless here, but do it for completeness

    snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm, t1a, inv_eff) # Meaningless here, but do it for completeness

    # Normalize SNR
    snr_norm_arr = snr_arr/np.amax(snr_arr, axis=0)
    snr_finitetm_corr_norm_arr = snr_finitetm_corr_arr/np.amax(snr_finitetm_corr_arr, axis=0)

    snr_norm_for_plot_arr = snr_norm_arr.T
    snr_finitetm_corr_arr_norm_for_plot_arr = snr_finitetm_corr_norm_arr.T

    # Find the lowest value that gets SNR to 95%
    first_fa_with_high_snr = fa_arr.ravel()[np.nonzero(snr_norm_for_plot_arr[4] > 0.95)[0][0]]

    # Plot

    grid_kwargs = dict(which="major", axis="both")
    x_label = r"$\alpha$ $\left[°\right]$"
    y_label = r"$SNR_{rel}$"
    x_lim = [np.amin(fa_arr), np.amax(fa_arr)]
    y_lim = None
    label_list = ["" for i in snr_finitetm_corr_arr_norm_for_plot_arr]
    legend_title = "Scenario"
    legend_kwargs = dict(loc='lower right', fancybox=True, shadow=False, framealpha=1.)
    colors = [None, "#D95319", None, "#77AC30", "#0072BD", "#77AC30", None, "#D95319", None]
    linestyles = ["", ":", "", ":", "-", ":", "", ":", ""]
    ax = basic_multiline_plot(fa_arr.ravel(), snr_finitetm_corr_arr_norm_for_plot_arr, label_list, #snr_norm_for_plot_arr
                              ax=None, figsize=figsize, colors=colors, linestyles=linestyles, alphas=None,
                              title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                              is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                              x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                              x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                              legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)
    #ax.axhline(0.95, color="gray", linestyle=":", linewidth=1)
    #ax.axvline(first_fa_with_high_snr, color="gray", linestyle=":", linewidth=1)

    # Define custom legend for aesthetic reasons
    legend_handles = [
        Line2D([0], [0], color="#0072BD", marker=None, linestyle="-", label="baseline"),
        Line2D([0], [0], color="#77AC30", marker=None, linestyle="--", label=r"vary $T_1$"),
        Line2D([0], [0], color="#D95319", marker=None, linestyle="--", label=r"vary $T_2$"),
    ]
    ax.legend(handles=legend_handles, framealpha=1.)

    plt.show()
    #######################################
    # Experiment: PSF SNR for variable TR #
    #######################################

    # Parameters
    fa = 95.  # 90
    tr = 7.65
    ti_arr = np.linspace(1100., 1500., num=401, endpoint=True).reshape((1, -1, 1)) #1100:20:1400 # [ms]
    td = 0.
    t_total = 100.*6*60*1000. # Do this to avoid discrete N_average phenomenons on the curve's appearance.

    # Calculate the array corresponding to readout time (tr - t_grad), which is the inverse of the pixelBW
    t_readout = (tr - t_grad)

    # Relaxation Parameter Variability
    vary_factor = 0.1
    t1a_arr = (1 + vary_factor * np.array([-1, 0, 1])) * t1a
    t2a_arr = (1 + vary_factor * np.array([-1, 0, 1])) * t2a

    total_t1a_arr, total_t2a_arr = np.meshgrid(t1a_arr, t2a_arr)
    total_t1a_arr = np.reshape(total_t1a_arr, (1, 1, -1))
    total_t2a_arr = np.reshape(total_t2a_arr, (1, 1, -1))

    # Total measurement time is time for inversion + rf pulses (including dummy) + dead time
    tm_arr = ti_arr + (n + n_dummy_tr) * tr + td

    # Calculate maximum number of averages (each average is 1 tag & 1 control image) without exceeding total acq. time
    n_avg_arr = np.floor(t_total / (2 * tm_arr) - n_dummy_tm)

    # Calculating PSFs
    # few points to make it computationally manageable. We only care about center here anyway.
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa, tr, ti_arr, total_t1a_arr, total_t2a_arr,
                                                       inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=1)
    z = z[:, 0]
    psf_max_arr = np.amax(psf, axis=0)  # Center of PSF is the max value in each case

    snr_arr = psf_max_arr * np.sqrt(n_avg_arr.reshape((-1, 1)) * t_readout)

    snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm_arr[0], t1a, inv_eff)

    # Normalize SNR
    snr_norm_arr = snr_arr / np.amax(snr_arr, axis=0)
    snr_finitetm_corr_norm_arr = snr_finitetm_corr_arr / np.amax(snr_finitetm_corr_arr, axis=0)

    snr_norm_for_plot_arr = snr_norm_arr.T
    snr_finitetm_corr_arr_norm_for_plot_arr = snr_finitetm_corr_norm_arr.T

    # Find the lowest value that gets SNR to 95%
    first_tr_with_high_snr = tr_arr.ravel()[np.nonzero(snr_norm_for_plot_arr[4] > 0.95)[0][0]]

    # Plot

    grid_kwargs = dict(which="major", axis="both")
    x_label = r"$TI$ $\left[ms\right]$"
    y_label = r"$SNR_{rel}$"
    x_lim = [np.amin(ti_arr), np.amax(ti_arr)]
    y_lim = None
    label_list = ["" for i in snr_finitetm_corr_arr_norm_for_plot_arr]
    legend_title = "Scenario"
    legend_kwargs = dict(loc='lower right', fancybox=True, shadow=False, framealpha=1.)
    colors = [None, "#D95319", None, "#77AC30", "#0072BD", "#77AC30", None, "#D95319", None]
    linestyles = ["", ":", "", ":", "-", ":", "", ":", ""]
    ax = basic_multiline_plot(ti_arr.ravel(), snr_finitetm_corr_arr_norm_for_plot_arr, label_list,
                              # snr_norm_for_plot_arr
                              ax=None, figsize=figsize, colors=colors, linestyles=linestyles, alphas=None,
                              title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs,
                              ticklabel_kwargs=None,
                              is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                              x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                              x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                              legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)
    # ax.axhline(0.95, color="gray", linestyle=":", linewidth=1)
    # ax.axvline(first_tr_with_high_snr, color="gray", linestyle=":", linewidth=1)

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