import numpy as np
from util import t1a_estimate, t2a_estimate, calculate_fwhm
from visualize import basic_multiline_plot
from fair_bssfp_simulation import simulate_bssfp_psf, simulate_fair_bssfp_signal_difference_psf

def main():
    # Plot parameters
    figsize = (4.416, 3.4)
    labels_fontsize = 11 #None #12

    # Parameter Optimization (Transient State)

    # Parameters

    b0       = 0.35                 # [T]
    n        = 96                   #         (number of pixels in x and y direction)
    bw_px_min = 130                  # [Hz/px] (minimum bandwidth per pixel)
    t_grad   = 2.7                  # [ms]    (dead time within TR for gradients/RF pulses)
    tr_max_bw  = 1e3/bw_px_min+t_grad     # [ms]    (maximum TR to assure high enough BW)
    n_dummy_tr  = 8                    #         (number of dummy rf pulses every bSSFP readout before signal is written)
    d_dummy_tr2 = 0                    #         (number of dummy inversion pulses before readout start)

    m0       = 1                    # [A/m]
    t1a      = t1a_estimate(b0, 2)   # [ms]
    t2a      = t2a_estimate(b0, 1)   # [ms]
    delta_f_max = 40                   # [Hz]    (maximum off-resonance in the brain)
    tr_max_offres = 1e3/(3*delta_f_max)           # [ms]    (maximum TR avoiding artefacts, Bieri et al.)
    delta_t  = 400                  # [ms]    (transit time)
    tau      = 1800                 # [ms]    (trailing time)
    partition_coef = 0.9                  # [ml/g]  (blood-tissue partition coefficient)
    inv_eff  = 0.8                  #         (inversion efficiency) # TODO: check why / if this value is chosen!!! # Maybe go for 0.98 instead
    q        = 1                    #         (correction factor (normally close to unity))
    perf     = 1                    #         (perfusion)

    tr_max    = min(tr_max_bw, tr_max_offres)

    #tscanmax = minutes(6) # TODO: Work with that



    ## Visualization
    ###########################################################################

    ## Singular Image

    fa_arr = np.linspace(60, 140, num=5, endpoint=True).reshape((1, -1)) #60:20:140
    tr  = 6
    ti  = 1100
    m   = m0 - 2 * m0 * np.exp(-ti/t1a)

    psf, z = simulate_bssfp_psf(m0, m, fa_arr, tr, t1a, t2a, n_dummy_tr, n)
    z = z[:, 0]
    psf = psf.T

    # Ok, this works! # TODO: check if I want to display the FWHM values
    grid_kwargs = dict(which="major", axis="both")
    x_label = "pixel coordinate"
    y_label = "PSF [A.U.]"
    x_lim = [-4, 4]
    label_list = [f"{fa}" for fa in fa_arr.ravel()]
    legend_title = "Excitation\nFlip Angle [°]"
    legend_kwargs = dict(loc='upper right', fancybox=True, shadow=False, framealpha=1.)
    colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F"]
    basic_multiline_plot(z, psf, label_list,
                         ax=None, figsize=figsize, colors=colors, linestyles=None, alphas=None,
                         title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=x_lim, y_lim=None, labels_fontsize=labels_fontsize,
                         legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)




    ## Difference Image (@FAs)

    fa_arr = np.linspace(60, 140, num=5, endpoint=True).reshape((1, -1)) #60:20:140
    n_dummy_tr_arr = np.around(fa_arr*0.05) # TODO: check this heuristic formula
    tr       = 6
    ti       = 1100

    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr, ti, t1a, t2a, inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr_arr, n_tr=n)
    z = z[:, 0]
    psf = psf.T


    grid_kwargs = dict(which="major", axis="both")
    x_label = "pixel coordinate"
    y_label = "PSF [A.U.]"
    x_lim = [-4, 4]
    label_list = [f"{fa}" for fa in fa_arr.ravel()]
    legend_title = "Excitation\nFlip Angle [°]"
    legend_kwargs = dict(loc='upper right', fancybox=True, shadow=False, framealpha=1.)
    colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F"]
    basic_multiline_plot(z, psf, label_list,
                         ax=None, figsize=figsize, colors=colors, linestyles=None, alphas=None,
                         title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=x_lim, y_lim=None, labels_fontsize=labels_fontsize,
                         legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)

    ## FWHM for FA

    fa_arr = np.linspace(1, 180, num=181, endpoint=True).reshape((1, -1)) #60:20:140
    n_dummy_tr = 8 # Here apparently we only look at 8. TODO: investigate

    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr, ti, t1a, t2a, inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=1000)
    z = z[:, 0]
    psf = psf.T

    fwhm_arr = np.array([calculate_fwhm(z, psf_individual)[0] for psf_individual in psf])

    grid_kwargs = dict(which="major", axis="both")
    x_label = "Excitation\nFlip Angle [°]"
    y_label = "FWHM [pixels]"
    label_list = [None]
    legend_title = None
    legend_kwargs = None
    colors = ["#0072BD"]
    basic_multiline_plot(fa_arr.ravel(), [fwhm_arr], label_list,
                         ax=None, figsize=figsize, colors=colors, linestyles=None, alphas=None,
                         title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=None, y_lim=None, labels_fontsize=labels_fontsize,
                         legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=False)

    ## Difference Image (@TRs)

    fa       = 90
    n_dummy_tr = 8
    tr_arr   = np.linspace(4., 6., num=3, endpoint=True).reshape((1, -1))
    ti       = 1100

    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa, tr_arr, ti, t1a, t2a, inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n)
    z = z[:, 0]
    psf = psf.T


    grid_kwargs = dict(which="major", axis="both")
    x_label = "pixel coordinate"
    y_label = "PSF [A.U.]"
    x_lim = [-4, 4]
    label_list = [f"{tr}" for tr in tr_arr.ravel()]
    legend_title = "Repetition\nTime [ms]"
    legend_kwargs = dict(loc='upper right', fancybox=True, shadow=False, framealpha=1.)
    colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F"]
    basic_multiline_plot(z, psf, label_list,
                         ax=None, figsize=figsize, colors=colors, linestyles=None, alphas=None,
                         title=None, x_label=x_label, y_label=y_label, grid_kwargs=grid_kwargs, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=x_lim, y_lim=None, labels_fontsize=labels_fontsize,
                         legend_title=legend_title, legend_kwargs=legend_kwargs, is_show=True)



    ## Optimization
    #######################################################################################

    # Parameters
    fa_arr      = np.linspace(80., 130., num=11, endpoint=True).reshape((1, -1, 1, 1, 1, 1)) #90:5:130     # [°]
    tr_arr      = np.linspace(5., 7., num=11, endpoint=True).reshape((1, 1, -1, 1, 1, 1)) #5:0.2:7      # [ms]
    ti_arr      = np.linspace(1100., 1400., num=16, endpoint=True).reshape((1, 1, 1, -1, 1, 1)) #1100:20:1400 # [ms]
    td_arr      = np.linspace(0., 400., num=5, endpoint=True).reshape((1, 1, 1, 1, -1, 1)) #0:100:400    # [ms]
    n_avg_arr   = np.linspace(60., 110., num=6, endpoint=True).reshape((1, 1, 1, 1, 1, -1)) #60:10:110

    tr2_arr = ti_arr + n * tr_arr + td_arr
    bw_factor = (tr_arr-t_grad)

    psf, z = simulate_fair_bssfp_signal_difference_psf(m0, fa_arr, tr_arr, ti_arr, t1a, t2a, inv_eff, perf, partition_coef,
                                                       delta_t, tau, q_func=None, n_dummy_tr=n_dummy_tr, n_tr=n,
                                                       n_points_psf=1000)
    z = z.reshape((-1,))
    psf_max = np.amax(psf, axis=0)
    """SNR_sweep  = nan(length(FAs),length(TRs),length(TIs),length(TDs),length(Navgs))
    FWHM_sweep = nan(length(FAs),length(TRs),length(TIs),length(TDs),length(Navgs))
    for j = 1:length(TIs)
      for i = 1:length(TDs)
        for b = 1:length(FAs)
          for c = 1:length(Navgs)
            for k = 1:length(TRs)
              if scantime(TRs(k),dummyTR,N,0,TIs(j),TDs(i),Navgs(c),dummyTR2) < tscanmax
                TR2                   = TIs(j)+N*TRs(k)+TDs(i)
                [PSFtemp,ztemp]       = FAIR_TrueFISP_PSF(M0,FAs(b),TRs(k),TIs(j),...
                                                          T1a,T2a,alpha,f,lambda,dt,tau,q,...
                                                          dummyTR,N)
                SNR_sweep(b,k,j,i,c)  = abs(PSFtemp(ztemp==0))...
                                          * sqrt( Navgs(c) * N*(TRs(k)-tgrad) )...
                                          * FiniteTMcorr(TR2,T1a)
                FWHM_sweep(b,k,j,i,c) = FormFWHM(PSFtemp,ztemp)
              end
            end
          end
        end
      end
    end

    # Find the SNR Maximum
    Nmax                    = 5
    [SNRmax,Imax]           = maxk(SNR_sweep(:),Nmax)
    [IFA,ITR,ITI,ITD,INavg] = ind2sub(size(SNR_sweep),Imax)

    # Print
    fprintf('\nMaximum SNR over multiple TR2:\n')
    fprintf(['  FA = %i° , TR = %1.1fms ,\n' ...
             '    TI = %ims , TD = %ims , Navg = %i\n'],...
            [FAs(IFA)',TRs(ITR)',...
             TIs(ITI)',TDs(ITD)',Navgs(INavg)']')

    # Find the FWHM Minimum
    Nmin                    = 5
    FWHM_red                = FWHM_sweep(:,:,1,1,1)
    [FWHMmin,Imin]          = mink(FWHM_red(:),Nmin)
    [IFA,ITR] = ind2sub(size(FWHM_red),Imin)

    # Print
    fprintf('\nMinimum FWHM over multiple TR2:\n')
    fprintf('  FA = %i° , TR = %1.1fms\n',...
            [FAs(IFA)',TRs(ITR)']')
    fprintf('\n')"""

if __name__ == "__main__":
    main()