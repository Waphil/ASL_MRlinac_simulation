import numpy as np

def simulate_fair_magnetization_difference(m0, inv_eff, perf, t, delta_t, tau, t1a, q_func=None):
    """
    Simulate the magnetization difference in FAIR between the control and label images. Implementation based on
    Martirosian et al. (2010) (DOI: 10.1007/s00259-010-1456-7)
    :param m0: float: equilibrium magnetization
    :param inv_eff: float: inversion efficiency
    :param perf: float: perfusion [ml/100 g/min]
    :param t: time point after inversion when signal is relevant [ms]
    :param delta_t: float: transit time [ms]
    :param tau: trailing time [ms]
    :param t1a: float: longitudinal relaxation time of arterial blood [ms]
    :param q_func: float or function where t is plugged in: correction factor (normally close to unity, which is the
    default)
    :return: float: magnetization difference between control and tag pulse in FAIR ASL experiment
    """
    if q_func is None: # If no q function is provided, assume that it is unity
        q_func = lambda t: 1.
    elif isinstance(q_func, (int, float, complex)):
        # If q_func is a number turn it into a function that returns just the number
        q_func = lambda t: q_func

    # Three different regimes as defined in Martirosian, et al. (2010) eq. (3)
    time_factor = np.zeros_like(t) # TODO: Adjust array size based on delta_t and tau
    np.putmask(time_factor, t <= (delta_t + tau), t - delta_t)
    np.putmask(time_factor, t > (delta_t + tau), tau)
    np.putmask(time_factor, t < delta_t, 0)
    return 2 * inv_eff * m0 * perf * time_factor * np.exp(-t/t1a) * q_func(t)


def simulate_fair_magnetization_difference_quipss2(m0, inv_eff, perf, partition_coef, ti, ti1, t1a):
    # Difference calculated by Martirosian, et al. (2010) eq. (4)
    return 2 * inv_eff * m0 * perf / partition_coef * ti1 * np.exp(-ti/t1a)

def simulate_bssfp_steady(m0, fa, tr, te, t1, t2, is_simple=False):
    """
    bSSFP Steady - State Signal calculation according to:
    - https: // mriquestions.com / true - fispfiesta.html
    - Bieri et al.(DOI 10.1002 / jmri.24163)
    :param m0: float: equilibrium magnetization
    :param fa: float: Flip angle [°]
    :param tr: float: repetition time [ms]
    :param te: float: echo time [ms]
    :param t1: float: longitudinal relaxation time of tissue [ms]
    :param t2: float: transversal relaxation time of tissue [ms]
    :param is_simple: (optional) bool: if true, use simplified version for TR << T1, T2
    :return: float: steady-state signal magnitude
    """
    if not is_simple:
        upper = (m0 * np.sin(np.deg2rad(fa)) * (1 - np.exp(-tr / t1)) * np.exp(-te / t2))
        lower = (1 - (np.exp(-tr / t1) - np.exp(-tr / t2)) * np.cos(np.deg2rad(fa)) - np.exp(-tr / t1) * np.exp(-tr / t2))
        return upper / lower
    else: #(simplified calculation if TR << T1, T2)
        upper = (m0 * np.sin(np.deg2rad(fa)) * np.exp(-te / t2))
        lower = (1 + np.cos(np.deg2rad(fa)) + (1 - np.cos(np.deg2rad(fa))) * (t1 / t2))
        return upper / lower

def simulate_bssfp_transient(m0, m, fa, tr, te, t1, t2, n_dummy_tr, n_tr, os_factor):
    """
    TrueFISP Transient Signal calculation according to:
     - Scheffler(DOI 10.1002/mrm.10421)
    :param m0: float: equilibrium longitudinal magnetization
    :param m: float: longitudinal magnetization at the first RF pulse of the bSSFP readout
    :param fa: float: Flip angle [°]
    :param tr: float: repetition time [ms]
    :param te: float: echo time [ms]
    :param t1: float: longitudinal relaxation time of tissue [ms]
    :param t2: float: transversal relaxation time of tissue [ms]
    :param n_dummy_tr: int: number of initial bSSFP rf pulses where no signal is written
    :param n_tr: int: number of bSSFP rf pulses where a signal is written
    :param os_factor: float: factor of oversampling (where 1 means 100% overampling)
    :return: float array of shape (n_tr,): transverse magnetization after each of the n_tr pulses where signal is read out
    """
    # Calculate steady state magnetization
    m_steadystate = simulate_bssfp_steady(m0, fa, tr, te, t1, t2, False)
    # Calculate decay constant (called lambda_1 in Scheffler's publication)
    decay_const = calculate_bssfp_decay_const(fa, tr, t1, t2)
    # Calculate different integer powers of the decay constant corresponding to different indexes of rf pulse.
    # Start at pulse (n_dummy_tr + 1) and then take (n_tr * (1+os_factor)) different numbers of pulses
    decay_const_power_arr = np.pow(decay_const, n_dummy_tr + 1 + np.arange(int((1+os_factor)* n_tr)))
    # Calculate the transient magnetization after each pulse # TODO: check if the array shapes are fine like this
    m_transient_arr = (np.sin(np.deg2rad(fa/2)) * m - m_steadystate) * decay_const_power_arr + m_steadystate
    return m_transient_arr

def calculate_bssfp_decay_const(fa, tr, t1, t2):
    """
    Calculate the decay constant of the transient bssfp signal. Defined as lambda_1 in
     - Scheffler(DOI 10.1002/mrm.10421)
    :param fa: float: Flip angle [°]
    :param tr: float: repetition time [ms]
    :param t1: float: longitudinal relaxation time of tissue [ms]
    :param t2: float: transversal relaxation time of tissue [ms]
    :return: float: decay rate lambda, with which the magnetization is multiplied every subsequent RF pulse during the
    transient phase of the bSSFP sequence.
    """
    t2_term = np.exp(-tr / t2) * np.pow(np.sin(np.deg2rad(fa/2)), 2)
    t1_term = np.exp(-tr / t1) * np.pow(np.cos(np.deg2rad(fa/2)), 2)
    return t2_term + t1_term

def simulate_bssfp_psf(m0, m, fa, tr, t1, t2, n_dummy_tr, n_tr):
    """
    TrueFISP PSF calculation according to:
    - Zhu & Qin (DOI 10.1016/j.mri.2022.01.015)
    :param m0: float: equilibrium longitudinal magnetization
    :param m: float: longitudinal magnetization at the first RF pulse of the bSSFP readout
    :param fa: float: Flip angle [°]
    :param tr: float: repetition time [ms]
    :param t1: float: longitudinal relaxation time of tissue [ms]
    :param t2: float: transversal relaxation time of tissue [ms]
    :param n_dummy_tr: int: number of initial bSSFP rf pulses where no signal is written
    :param n_tr: int: number of bSSFP rf pulses where a signal is written
    :return: float array of shape (100*n_tr + 1,): point spread function of
    """
    # Define pixel coordinates for PSF
    z = np.linspace(-n_tr/2, n_tr/2, 100*n_tr+1, endpoint=True)

    # Try to adjust the shape of z to be compatible with input parameter shapes that are multi-dimensional.
    # First axis of input parameters has to be length 1 to allow for the z-axis to go there
    ndim_combined_arr = np.ndim(m0 + m + fa + tr + t1 + t2 + n_dummy_tr + n_tr)
    if ndim_combined_arr > 1:
        z = np.reshape(z, tuple([-1] + [1 for axis in range(ndim_combined_arr-1)]))

    # Calculate steady state magnetization
    m_steadystate = simulate_bssfp_steady(m0, fa, tr, tr/2., t1, t2, False) # Assume TE = TR/2
    # Calculate decay constant (called lambda in Zhu & Qin's publication)
    decay_const = calculate_bssfp_decay_const(fa, tr, t1, t2)

    fraction_term = calculate_centric_bssfp_psf_large_fraction(z, decay_const, n_tr)

    multiplication_term = np.pow(decay_const, n_dummy_tr) * (m * np.sin(np.deg2rad(fa/2))-m_steadystate)

    addition_term = m_steadystate * np.sinc(z) # Sinc is normalized, so it contains pi.

    return (fraction_term * multiplication_term + addition_term), z

def calculate_centric_bssfp_psf_large_fraction(z, decay_const, n_tr):
    """
    Calculates the large fraction term that appears for the PSF of centric ordered bSSFP.
    See Zhu & Qin (DOI 10.1016/j.mri.2022.01.015) equation 42.
    :param z: array of floats: pixel coordinate in real space.
    :param decay_const: float: decay rate lambda, with which the magnetization is multiplied every subsequent RF pulse during the
    transient phase of the bSSFP sequence.
    :param n_tr: int: number of bSSFP rf pulses where a signal is written
    :return: array of floats in same shape as z: fraction term for PSF.
    """
    fraction_upper_term_1 = np.pow(decay_const, n_tr) * n_tr * np.log(decay_const) * np.cos(np.pi*z)
    fraction_upper_term_2 = np.pow(decay_const, n_tr) * np.pi * z * np.sin(np.pi*z)
    fraction_upper_term_3 = - n_tr * np.log(decay_const)
    fraction_upper_term = fraction_upper_term_1 + fraction_upper_term_2 + fraction_upper_term_3

    fraction_lower_term = np.pow(n_tr * np.log(decay_const), 2) + np.pow(np.pi * z, 2)

    fraction_term = fraction_upper_term / fraction_lower_term

    if np.count_nonzero(fraction_lower_term == 0.):  # Fix the divide by 0 error at z=0 when lambda is 1.
        where_zero_over_zero_arr = (fraction_upper_term == 0.) & (fraction_lower_term == 0.)
        fraction_term[where_zero_over_zero_arr] = 1.

    return fraction_term

def simulate_fair_bssfp_signal_difference(m0, fa, tr, ti, t1a, t2a, inv_eff, perf, delta_t, tau,
                                          q_func=None, n_dummy_tr=0):
    """
    Calculates difference of arterial blood signal between control and tag. Uses PASL and bSSFP signal functions.
    Does not use the PSF approach
    :param m0: float: equilibrium magnetization
    :param fa: float: Flip angle [°]
    :param tr: float: repetition time [ms]
    :param te: float: echo time [ms]
    :param ti: float: inversion time [ms]
    :param ti1: float: saturation pulse time [ms]
    :param t1a: float: longitudinal relaxation time of arterial blood [ms]
    :param t2a: float: transversal relaxation time of arterial blood [ms]
    :param inv_eff: float: inversion efficiency
    :param perf: float: perfusion [ml/100 g/min]
    :param delta_t: float: transit time [ms]
    :param tau: trailing time [ms]
    :param q_func: float or function where t is plugged in: correction factor (normally close to unity, which is the
    default)
    :return:
    """
    # Calculate decay constant (called lambda in Zhu & Qin's publication)
    decay_const = calculate_bssfp_decay_const(fa, tr, t1a, t2a)
    # Calculate the longitudinal magnetization difference between tag and control at the start of the bSSFP readout
    fair_magnetization_difference = simulate_fair_magnetization_difference(m0, inv_eff, perf, ti, delta_t, tau, t1a,
                                                                           q_func=q_func)

    multiplication_term = np.pow(decay_const, n_dummy_tr) * np.sin(np.deg2rad(fa/2)) * fair_magnetization_difference

    return multiplication_term

def simulate_fair_bssfp_signal_difference_psf(m0, fa, tr, ti, t1a, t2a, inv_eff, perf, delta_t, tau,
                                              q_func=None, n_dummy_tr=0, n_tr=96, n_points_psf=None):
    """
    Calculates difference of arterial blood signal between control and tag. Uses PASL and bSSFP signal functions.
    Estimates the signal difference as the center of the PSF.
    :param m0: float: equilibrium magnetization
    :param fa: float: Flip angle [°]
    :param tr: float: repetition time [ms]
    :param ti: float: inversion time [ms]
    :param t1a: float: longitudinal relaxation time of arterial blood [ms]
    :param t2a: float: transversal relaxation time of arterial blood [ms]
    :param inv_eff: float: inversion efficiency
    :param perf: float: perfusion [ml/100 g/min]
    :param delta_t: float: transit time [ms]
    :param tau: trailing time [ms]
    :param q_func: float or function where ti is plugged in: correction factor (normally close to unity, which is the
    default)
    :param n_dummy_tr: int: number of initial bSSFP rf pulses where no signal is written
    :param n_tr: int: number of bSSFP rf pulses where a signal is written
    :param n_points_psf: (optional) int: number of points of PSF that get simulated. 100 by default. For a reliable
    FWHM you need more
    :return:
    """
    if n_points_psf is None:
        n_points_psf = 100*n_tr+1

    # Define pixel coordinates for PSF
    z = np.linspace(-n_tr/2, n_tr/2, n_points_psf, endpoint=True)

    # Try to adjust the shape of z to be compatible with input parameter shapes that are multi-dimensional.
    # First axis of input parameters has to be length 1 to allow for the z-axis to go there
    ndim_combined_arr = np.ndim(m0 + fa + tr + ti + t1a + t2a + n_dummy_tr + n_tr)
    if ndim_combined_arr > 1:
        z = np.reshape(z, tuple([-1] + [1 for axis in range(ndim_combined_arr-1)]))

    # Calculate decay constant (called lambda in Zhu & Qin's publication)
    decay_const = calculate_bssfp_decay_const(fa, tr, t1a, t2a)
    # Calculate the longitudinal magnetization difference between tag and control at the start of the bSSFP readout
    fair_magnetization_difference = simulate_fair_magnetization_difference(m0, inv_eff, perf, ti, delta_t, tau, t1a,
                                                                           q_func=q_func)

    fraction_term = calculate_centric_bssfp_psf_large_fraction(z, decay_const, n_tr)

    multiplication_term = np.pow(decay_const, n_dummy_tr) * np.sin(np.deg2rad(fa/2)) * fair_magnetization_difference

    return (fraction_term * multiplication_term), z
