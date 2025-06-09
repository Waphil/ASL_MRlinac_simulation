import numpy as np
from util import t1a_estimate, t2a_estimate, calculate_fwhm
from visualize import basic_multiline_plot
from fair_bssfp_simulation import (simulate_bssfp_psf, simulate_fair_bssfp_signal_difference_psf,
                                   simulate_fair_bssfp_signal_difference, correction_factor_for_finite_tm)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Creating figures for the publication
##################################################################


def main():
    # Plot parameters
    figsize = (4.416, 3.4)
    figsize_psf = (3.3, 3.4)#(2.93, 3.4)
    labels_fontsize = 11
    fwhm_line_height = 30.
    is_show_sinc_on_psf = True # Decide if we show the ideal sinc on the PSF
    is_show_sinc_on_fwhm = True # Decide if we show the FWHM of the ideal sinc on the FWHM plot
    is_show_snr_eff = True # Decide if we show SNR efficiency (True) or absolute SNR (False)
    is_show_snr_with_const_tm = True # Decide if we want to show the effSNR for constant TM in TI and TR plots

    # Hardware / constants
    b0       = 0.35                   # [T]
    bw_px_min = 130                   # [Hz/px] (minimum bandwidth per pixel)
    t_grad   = 3.43 #2.7              # [ms]    (dead time within TR for gradients/RF pulses)
    tr_max_bw  = 1e3/bw_px_min+t_grad # [ms]    (maximum TR to assure high enough BW)
    delta_f_max = 40.                 # [Hz]    (maximum off-resonance in the brain at 0.35 T)
    if b0 > 0.35:
        delta_f_max = 66.#140?        # [Hz]    (maximum off-resonance in the brain at 3 T(?) Simply chose value that allows reasonable TR)
    tr_max_offres = 1e3/(3*delta_f_max) # [ms]    (maximum TR avoiding artefacts, Bieri et al.)
    tr_max    = min(tr_max_bw, tr_max_offres)

    # Image settings constants
    n        = 96                   #         (number of pixels in x and y direction)
    t_total  = 6*60*1000.           # [ms]    (total acquisition time constraint set for whole sequence in optimization)
    is_optimize_based_on_psf = True #         (choose false if you want to optimize based on the k-space center instead)

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
    fa_arr      = np.linspace(50., 150., num=101, endpoint=True).reshape((1, -1, 1, 1, 1)) #90:5:130     # [°]
    tr_arr      = np.linspace(4.5, 8.3, num=77, endpoint=True).reshape((1, 1, -1, 1, 1)) #5:0.2:7      # [ms]
    ti_arr      = np.linspace(900., 1500., num=121, endpoint=True).reshape((1, 1, 1, -1, 1)) #1100:20:1400 # [ms]
    td_arr      = np.linspace(0., 800., num=81, endpoint=True).reshape((1, 1, 1, 1, -1)) #0:100:400    # [ms]

    # Make sure to only simulate tr below the maximum allowed
    tr_arr = tr_arr[tr_arr <= tr_max].reshape((1, 1, -1, 1, 1))
    if np.size(tr_arr) == 0:
        raise ValueError("The chosen configuration has no valid TRs, as the minimum simulated is above maximum allowed.")

    # Total measurement time is time for inversion + rf pulses (including dummy) + dead time
    tm_arr = ti_arr + (n + n_dummy_tr) * tr_arr + td_arr

    # Calculate maximum number of averages (each average is 1 tag & 1 control image) without exceeding total acq. time
    #n_avg_arr = np.floor(t_total / (2*tm_arr) - n_dummy_tm)

    # Calculate the array corresponding to readout time (tr - t_grad), which is the inverse of the pixelBW
    t_readout_arr = (tr_arr-t_grad)

    if is_optimize_based_on_psf:
        psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr_arr, ti_arr, t1a, t2a, inv_eff, perf,
                                                           partition_coef, delta_t, tau, q_func=None,
                                                           n_dummy_tr=n_dummy_tr, n_tr=n, n_points_psf=3)
        z = z.reshape((-1,))
        psf_max = np.amax(psf, axis=0)
    else:
        # Need to call it max psf to be compatible with
        psf_max = simulate_fair_bssfp_signal_difference(m0, fa_arr, tr_arr, ti_arr, t1a, t2a, inv_eff, perf,
                                                        partition_coef, delta_t, tau, q_func=None,
                                                        n_dummy_tr=n_dummy_tr)
        psf_max = psf_max[0]

    snr_arr = psf_max * np.sqrt(t_readout_arr[0])  # Have to remove first dimension readout to match them
    snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm_arr[0], t1a, inv_eff)
    snr_eff_finitetm_corr_arr = snr_finitetm_corr_arr / np.sqrt(tm_arr[0])

    best_settings_indices_masks = [
        np.count_nonzero(snr_eff_finitetm_corr_arr == np.amax(snr_eff_finitetm_corr_arr),
                         axis=tuple([j for j in range(snr_eff_finitetm_corr_arr.ndim) if j != i]))
        for i in range(snr_eff_finitetm_corr_arr.ndim)]
    best_settings_indices = [np.argmax(best_settings) for best_settings in best_settings_indices_masks]

    fa_optimum = fa_arr.ravel()[best_settings_indices[0]]
    tr_optimum = tr_arr.ravel()[best_settings_indices[1]]
    ti_optimum = ti_arr.ravel()[best_settings_indices[2]]
    td_optimum = td_arr.ravel()[best_settings_indices[3]]
    tm_optimum = tm_arr[0,0][*best_settings_indices[1:]]
    n_avg_optimum = np.floor(t_total / (2*tm_optimum) - n_dummy_tm)

    print(f"-----------------------\n"
          f"Optimized settings:\n"
          f"FA = {fa_optimum}°\n"
          f"TR = {tr_optimum:.2f} ms\n"
          f"TI = {ti_optimum} ms\n"
          f"TD = {td_optimum} ms\n"
          f"TM = {tm_optimum} ms\n"
          f"N_avg = {n_avg_optimum}\n")
    print(f"The TM correction factor for those settings is "
          f"{correction_factor_for_finite_tm(tm_arr[0], t1a, inv_eff)[0][*best_settings_indices[1:]]:.4f}")

    ########################################
    # Experiment: show PSF for Variable TR #
    ########################################

    fa       = fa_optimum #95.
    n_dummy_tr  = 8
    tr_arr   = np.linspace(5., 8., num=4, endpoint=True).reshape((1, -1))
    ti       = ti_optimum #1345.
    td       = td_optimum #0.

    # Calculating PSFs
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa, tr_arr, ti, t1a, t2a, inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=None)
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
    linestyles = ["-", "-", "-", "-", "-", "-", "-"]
    if is_show_sinc_on_psf:
        psf_sinc = np.sinc(z).reshape((1, -1))
        overall_psf_max = np.amax(psf)
        psf_sinc_normalized = psf_sinc / np.amax(psf_sinc) * overall_psf_max
        psf = np.append(psf, psf_sinc_normalized, axis=0)
        y_lim[0] = min(y_lim[0], 1.1*np.amin(psf_sinc_normalized))
        label_list += ["Ideal\nSinc"]
        colors = colors[:(len(label_list)-1)]
        colors += ["darkgray"]
        linestyles = linestyles[:(len(label_list)-1)]
        linestyles += ["--"]
    ax = basic_multiline_plot(z, psf, label_list,
                         ax=None, figsize=figsize_psf, colors=colors, linestyles=linestyles, alphas=None,
                         title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                         legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)
    for psf_individual, fwhm_tuple, color in zip(psf, fwhm_tuple_list, colors):
        x_coords = np.array([z[fwhm_tuple[1]], z[fwhm_tuple[2]]])
        y_coords = np.array([psf_individual[fwhm_tuple[1]], psf_individual[fwhm_tuple[2]]])
        ax.errorbar(x_coords, y_coords, yerr=fwhm_line_height, ecolor=color, fmt="", linestyle="")

    ########################################
    # Experiment: show PSF for Variable FA #
    ########################################

    fa_arr = np.linspace(70, 130, num=4, endpoint=True).reshape((1, -1))
    n_dummy_tr_arr = 8.
    tr       = tr_optimum #7.65
    ti       = ti_optimum #1345.
    td       = td_optimum #0.

    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr, ti, t1a, t2a, inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr_arr, n_tr=n,
                                                       n_points_psf=None)
    z = z[:, 0]
    psf = psf.T

    fwhm_tuple_list = [calculate_fwhm(z, psf_individual) for psf_individual in psf]

    grid_kwargs = dict(which="major", axis="both")
    x_label = r"pixel coordinate $z$"
    y_label = r"$PSF$ $\left[A.U.\right]$"
    x_lim = [-3.5, 3.5]
    y_lim = [-10., 115.]
    label_list = [f"{fa}" for fa in fa_arr.ravel()]
    legend_title = f"$TR$ = {tr:.1f} ms,\n$\\alpha$ $\\left[°\\right]$"
    legend_kwargs = dict(loc='upper right', fancybox=True, shadow=False, framealpha=1.)
    colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F"]
    linestyles = ["-", "-", "-", "-", "-", "-", "-"]
    if is_show_sinc_on_psf:
        psf_sinc = np.sinc(z).reshape((1, -1))
        overall_psf_max = np.amax(psf)
        psf_sinc_normalized = psf_sinc / np.amax(psf_sinc) * overall_psf_max
        psf = np.append(psf, psf_sinc_normalized, axis=0)
        y_lim[0] = min(y_lim[0], 1.1*np.amin(psf_sinc_normalized))
        label_list += ["Ideal\nSinc"]
        colors = colors[:(len(label_list)-1)]
        colors += ["darkgray"]
        linestyles = linestyles[:(len(label_list)-1)]
        linestyles += ["--"]
    ax = basic_multiline_plot(z, psf, label_list,
                         ax=None, figsize=figsize_psf, colors=colors, linestyles=linestyles, alphas=None,
                         title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                         legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)
    for psf_individual, fwhm_tuple, color in zip(psf, fwhm_tuple_list, colors):
        x_coords = np.array([z[fwhm_tuple[1]], z[fwhm_tuple[2]]])
        y_coords = np.array([psf_individual[fwhm_tuple[1]], psf_individual[fwhm_tuple[2]]])
        ax.errorbar(x_coords, y_coords, yerr=fwhm_line_height, ecolor=color, fmt="", linestyle="")

    #######################################
    # Experiment: PSF SNR for variable TR #
    #######################################

    # Parameters
    fa   = fa_optimum #95.
    tr_arr = np.linspace(4.5, 8.3, num=236, endpoint=True).reshape((1, -1, 1)) # 4:0.01:9.2
    ti   = ti_optimum #1345.
    td   = td_optimum #0.

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
    #tm_arr = np.maximum(tm_arr, np.amax(tm_arr)) # Experiment: show same TM for all images (the max)

    # Calculating PSFs
    # few points to make it computationally manageable. We only care about center here anyway.
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa, tr_arr, ti, total_t1a_arr, total_t2a_arr,
                                                       inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=3)
    z = z[:, 0]
    psf_max_arr = np.amax(psf, axis=0) # Center of PSF is the max value in each case

    snr_arr = psf_max_arr * np.sqrt(t_readout_arr.reshape((-1, 1))) # assumes constant TM
    #snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm_arr[0], t1a, inv_eff)
    snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm_arr[0], total_t1a_arr[0], inv_eff)
    snr_eff_finitetm_corr_arr = snr_finitetm_corr_arr / np.sqrt(tm_arr[0])

    # Normalize SNR
    snr_norm_arr = snr_arr/np.amax(snr_arr, axis=0)
    snr_finitetm_corr_norm_arr = snr_finitetm_corr_arr/np.amax(snr_finitetm_corr_arr, axis=0)
    snr_eff_finitetm_corr_norm_arr = snr_eff_finitetm_corr_arr/np.amax(snr_eff_finitetm_corr_arr, axis=0)

    snr_norm_for_plot_arr = snr_norm_arr.T
    if is_show_snr_eff:
        snr_plot_arr = snr_eff_finitetm_corr_norm_arr.T
    else:
        snr_plot_arr = snr_finitetm_corr_norm_arr.T

    # Plot

    grid_kwargs = dict(which="major", axis="both")
    x_label = r"$TR$ $\left[ms\right]$"
    y_label = r"relative $effSNR_{PSF}$" if is_show_snr_eff else r"relative $SNR_{PSF}$"
    x_lim = [np.amin(tr_arr), np.amax(tr_arr)]
    y_lim = None
    label_list = ["" for i in snr_plot_arr]
    legend_title = "Scenario"
    legend_kwargs = dict(loc='lower right', fancybox=True, shadow=False, framealpha=1.)
    colors = [None, "#D95319", None, "#77AC30", "#0072BD", "#77AC30", None, "#D95319", None]
    linestyles = ["", ":", "", ":", "-", ":", "", ":", ""]
    if is_show_snr_with_const_tm:
        # Add normalized SNR with constant TM if desired
        snr_plot_arr = np.append(snr_plot_arr, snr_norm_for_plot_arr[[4]], axis=0)
        label_list += [""]
        colors += ["#30A2ED"]
        linestyles += ["--"]
    ax = basic_multiline_plot(tr_arr.ravel(), snr_plot_arr, label_list,
                              ax=None, figsize=figsize, colors=colors, linestyles=linestyles, alphas=None,
                              title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                              is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                              x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                              x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                              legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)

    # Define custom legend for aesthetic reasons
    legend_handles = [
        Line2D([0], [0], color="#0072BD", marker=None, linestyle="-", label="baseline"),
        Line2D([0], [0], color="#77AC30", marker=None, linestyle=":",
               label=f"vary $T_1$ by ±{vary_factor:.0%}"),
        Line2D([0], [0], color="#D95319", marker=None, linestyle=":",
               label=f"vary $T_2$ by ±{vary_factor:.0%}"),
    ]
    if is_show_snr_with_const_tm:
        legend_handles.append(Line2D([0], [0], color="#30A2ED", marker=None, linestyle="--",
                                     label="baseline (constant TM)"))
    ax.legend(handles=legend_handles, framealpha=1.)

    #######################################
    # Experiment: PSF SNR for variable FA #
    #######################################

    # Parameters
    fa_arr   = np.linspace(45., 155., num=111, endpoint=True).reshape((1, -1, 1)) # 45:1:155
    tr = tr_optimum #7.65
    ti   = ti_optimum #1345.
    td = td_optimum #0.
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

    # Calculating PSFs
    # few points to make it computationally manageable. We only care about center here anyway.
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr, ti, total_t1a_arr, total_t2a_arr,
                                                       inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=3)
    z = z[:, 0]
    psf_max_arr = np.amax(psf, axis=0) # Center of PSF is the max value in each case

    snr_arr = psf_max_arr * np.sqrt(t_readout) # Meaningless here, but do it for completeness
    #snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm, t1a, inv_eff) # Meaningless here, but do it for completeness
    snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm, total_t1a_arr[0], inv_eff)
    snr_eff_finitetm_corr_arr = snr_finitetm_corr_arr / np.sqrt(tm)

    # Normalize SNR
    snr_norm_arr = snr_arr/np.amax(snr_arr, axis=0)
    snr_finitetm_corr_norm_arr = snr_finitetm_corr_arr/np.amax(snr_finitetm_corr_arr, axis=0)
    snr_eff_finitetm_corr_norm_arr = snr_eff_finitetm_corr_arr/np.amax(snr_eff_finitetm_corr_arr, axis=0)

    snr_norm_for_plot_arr = snr_norm_arr.T
    if is_show_snr_eff:
        snr_plot_arr = snr_eff_finitetm_corr_norm_arr.T
    else:
        snr_plot_arr = snr_finitetm_corr_norm_arr.T

    # Plot

    grid_kwargs = dict(which="major", axis="both")
    x_label = r"$\alpha$ $\left[°\right]$"
    y_label = r"relative $effSNR_{PSF}$" if is_show_snr_eff else r"relative $SNR_{PSF}$"
    x_lim = [np.amin(fa_arr), np.amax(fa_arr)]
    y_lim = None
    label_list = ["" for i in snr_plot_arr]
    legend_title = "Scenario"
    legend_kwargs = dict(loc='lower right', fancybox=True, shadow=False, framealpha=1.)
    colors = [None, "#D95319", None, "#77AC30", "#0072BD", "#77AC30", None, "#D95319", None]
    linestyles = ["", ":", "", ":", "-", ":", "", ":", ""]
    ax = basic_multiline_plot(fa_arr.ravel(), snr_plot_arr, label_list, #snr_norm_for_plot_arr
                              ax=None, figsize=figsize, colors=colors, linestyles=linestyles, alphas=None,
                              title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                              is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                              x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                              x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                              legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)

    # Define custom legend for aesthetic reasons
    legend_handles = [
        Line2D([0], [0], color="#0072BD", marker=None, linestyle="-", label="baseline"),
        Line2D([0], [0], color="#77AC30", marker=None, linestyle=":",
               label=f"vary $T_1$ by ±{vary_factor:.0%}"),
        Line2D([0], [0], color="#D95319", marker=None, linestyle=":",
               label=f"vary $T_2$ by ±{vary_factor:.0%}"),
    ]
    ax.legend(handles=legend_handles, framealpha=1.)

    #######################################
    # Experiment: PSF SNR for variable TI #
    #######################################

    # Parameters
    fa = fa_optimum #95.
    tr = tr_optimum #7.65
    ti_arr = np.linspace(900., 1500., num=601, endpoint=True).reshape((1, -1, 1)) #1100:20:1400 # [ms]
    td = td_optimum #0.

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
    #tm_arr = np.maximum(tm_arr, np.amax(tm_arr)) # Experiment: show same TM for all images (the max)

    # Calculating PSFs
    # few points to make it computationally manageable. We only care about center here anyway.
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa, tr, ti_arr, total_t1a_arr, total_t2a_arr,
                                                       inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=3)
    z = z[:, 0]
    psf_max_arr = np.amax(psf, axis=0)  # Center of PSF is the max value in each case

    snr_arr = psf_max_arr * np.sqrt(t_readout)
    #snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm_arr[0], t1a, inv_eff)
    snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm_arr[0], total_t1a_arr[0], inv_eff)
    snr_eff_finitetm_corr_arr = snr_finitetm_corr_arr / np.sqrt(tm_arr[0])

    # Normalize SNR
    snr_norm_arr = snr_arr / np.amax(snr_arr, axis=0)
    snr_finitetm_corr_norm_arr = snr_finitetm_corr_arr / np.amax(snr_finitetm_corr_arr, axis=0)
    snr_eff_finitetm_corr_norm_arr = snr_eff_finitetm_corr_arr / np.amax(snr_eff_finitetm_corr_arr, axis=0)

    snr_norm_for_plot_arr = snr_norm_arr.T
    if is_show_snr_eff:
        snr_plot_arr = snr_eff_finitetm_corr_norm_arr.T
    else:
        snr_plot_arr = snr_finitetm_corr_norm_arr.T

    # Plot

    grid_kwargs = dict(which="major", axis="both")
    x_label = r"$TI$ $\left[ms\right]$"
    y_label = r"relative $effSNR_{PSF}$" if is_show_snr_eff else r"relative $SNR_{PSF}$"
    x_lim = [np.amin(ti_arr), np.amax(ti_arr)]
    y_lim = None
    label_list = ["" for i in snr_plot_arr]
    legend_title = "Scenario"
    legend_kwargs = dict(loc='lower right', fancybox=True, shadow=False, framealpha=1.)
    colors = [None, "#D95319", None, "#77AC30", "#0072BD", "#77AC30", None, "#D95319", None]
    linestyles = ["", ":", "", ":", "-", ":", "", ":", ""]
    if is_show_snr_with_const_tm:
        # Add normalized SNR with constant TM if desired
        snr_plot_arr = np.append(snr_plot_arr, snr_norm_for_plot_arr[[4]], axis=0)
        label_list += [""]
        colors += ["#30A2ED"]
        linestyles += ["--"]
    ax = basic_multiline_plot(ti_arr.ravel(), snr_plot_arr, label_list,
                              # snr_norm_for_plot_arr
                              ax=None, figsize=figsize, colors=colors, linestyles=linestyles, alphas=None,
                              title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs,
                              ticklabel_kwargs=None,
                              is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                              x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                              x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                              legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)

    # Define custom legend for aesthetic reasons
    legend_handles = [
        Line2D([0], [0], color="#0072BD", marker=None, linestyle="-", label="baseline"),
        Line2D([0], [0], color="#77AC30", marker=None, linestyle=":",
               label=f"vary $T_1$ by ±{vary_factor:.0%}"),
        Line2D([0], [0], color="#D95319", marker=None, linestyle=":",
               label=f"vary $T_2$ by ±{vary_factor:.0%}"),
    ]
    if is_show_snr_with_const_tm:
        legend_handles.append(Line2D([0], [0], color="#30A2ED", marker=None, linestyle="--",
                                     label="baseline (constant TM)"))
    ax.legend(handles=legend_handles, framealpha=1.)

    #######################################
    # Experiment: PSF SNR for variable TM #
    #######################################

    # Parameters
    fa = fa_optimum #95.
    tr = tr_optimum #7.65
    ti = ti_optimum #1345.
    td_arr = np.linspace(0., 800., num=801, endpoint=True).reshape((1, -1, 1))

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
    tm_arr = ti + (n + n_dummy_tr) * tr + td_arr
    #tm_arr = np.maximum(tm_arr, np.amax(tm_arr)) # Experiment: show same TM for all images (the max)

    # Calculating PSFs
    # few points to make it computationally manageable. We only care about center here anyway.
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa, tr, ti, total_t1a_arr, total_t2a_arr,
                                                       inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=3)
    z = z[:, 0]
    psf_max_arr = np.amax(psf, axis=0)  # Center of PSF is the max value in each case

    snr_arr = psf_max_arr * np.sqrt(t_readout)
    snr_finitetm_corr_arr = snr_arr * correction_factor_for_finite_tm(tm_arr[0], total_t1a_arr[0], inv_eff)
    snr_eff_finitetm_corr_arr = snr_finitetm_corr_arr / np.sqrt(tm_arr[0])

    # Normalize SNR
    snr_norm_arr = snr_arr / np.amax(snr_arr, axis=0)
    snr_finitetm_corr_norm_arr = snr_finitetm_corr_arr / np.amax(snr_finitetm_corr_arr, axis=0)
    snr_eff_finitetm_corr_norm_arr = snr_eff_finitetm_corr_arr / np.amax(snr_eff_finitetm_corr_arr, axis=0)

    snr_norm_for_plot_arr = snr_norm_arr.T
    if is_show_snr_eff:
        snr_plot_arr = snr_eff_finitetm_corr_norm_arr.T
    else:
        snr_plot_arr = snr_finitetm_corr_norm_arr.T

    # Plot

    grid_kwargs = dict(which="major", axis="both")
    x_label = r"$TM$ $\left[ms\right]$"
    y_label = r"relative $effSNR_{PSF}$" if is_show_snr_eff else r"relative $SNR_{PSF}$"
    x_lim = [np.amin(tm_arr), np.amax(tm_arr)]
    y_lim = None
    label_list = ["" for i in snr_plot_arr]
    legend_title = "Scenario"
    legend_kwargs = dict(loc='lower right', fancybox=True, shadow=False, framealpha=1.)
    colors = [None, "#D95319", None, "#77AC30", "#0072BD", "#77AC30", None, "#D95319", None]
    linestyles = ["", ":", "", ":", "-", ":", "", ":", ""]
    ax = basic_multiline_plot(tm_arr.ravel(), snr_plot_arr, label_list,
                              # snr_norm_for_plot_arr
                              ax=None, figsize=figsize, colors=colors, linestyles=linestyles, alphas=None,
                              title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs,
                              ticklabel_kwargs=None,
                              is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                              x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                              x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                              legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)

    # Define custom legend for aesthetic reasons
    legend_handles = [
        Line2D([0], [0], color="#0072BD", marker=None, linestyle="-", label="baseline"),
        Line2D([0], [0], color="#77AC30", marker=None, linestyle=":",
               label=f"vary $T_1$ by ±{vary_factor:.0%}"),
        Line2D([0], [0], color="#D95319", marker=None, linestyle=":",
               label=f"vary $T_2$ by ±{vary_factor:.0%}"),
    ]
    ax.legend(handles=legend_handles, framealpha=1.)

    #############################################
    # Experiment: PSF FWHM for variable FA & TR #
    #############################################

    # Parameters
    fa_arr   = np.linspace(45., 155., num=111, endpoint=True).reshape((1, -1, 1))
    tr_arr   = np.linspace(5., 8., num=4, endpoint=True).reshape((1, 1, -1))
    ti   = 1345. #1300

    # Calculating PSFs
    # Need many points because now we care about width of PSF
    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr_arr, ti, t1a, t2a,
                                                       inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=1000*n)
    z = z[:, 0]
    psf = psf.T

    fwhm_arr = np.array([[calculate_fwhm(z, psf_fa)[0][0] for psf_fa in psf_tr] for psf_tr in psf])

    # Plot settings
    grid_kwargs = dict(which="major", axis="both")
    x_label = r"$\alpha$ $\left[°\right]$"
    y_label = r"$FWHM$ of $PSF$ $\left[voxels\right]$"
    x_lim = [np.amin(fa_arr), np.amax(fa_arr)]
    y_lim = None
    label_list = [f"{tr}" for tr in tr_arr.ravel()]
    legend_title = f"$TR$ $\\left[ms\\right]$"
    legend_kwargs = dict(loc='upper left', fancybox=True, shadow=False, framealpha=1.)
    colors = ["#0072BD","#D95319","#EDB120","#7E2F8E"]
    linestyles = ["-", "-", "-", "-"]
    if is_show_sinc_on_fwhm:
        psf_sinc = np.sinc(z)
        ideal_sinc_fwhm = calculate_fwhm(z, psf_sinc)[0][0]*np.ones((1, np.size(fa_arr)))
        fwhm_arr = np.append(fwhm_arr, ideal_sinc_fwhm, axis=0)
        label_list += ["Ideal Sinc"]
        colors = colors[:(len(label_list)-1)]
        colors += ["darkgray"]
        linestyles = linestyles[:(len(label_list)-1)]
        linestyles += ["--"]
    ax = basic_multiline_plot(fa_arr.ravel(), fwhm_arr, label_list,
                         ax=None, figsize=figsize_psf, colors=colors, linestyles=linestyles, alphas=None,
                         title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=x_lim, y_lim=y_lim, labels_fontsize=labels_fontsize,
                         legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)

    plt.show()

if __name__ == "__main__":
    main()