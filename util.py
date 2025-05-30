import numpy as np

def t1a_estimate(b0, model_id):
    """
    Arterial blood T1 calculation depending on field strength according to :
    - Zhang et al. (DOI 10.1002/mrm.24550)
    - Rooney et al. (DOI 10.1002/mrm.21122)
    - Hales et al. (DOI 10.1177/0271678X15605856)
    :param b0: float: magnetic field strength [T]
    :param model_id: int: slected model. 1 for Zhang, 2 for Rooney, 3 for Hales
    :return: T1 of arterial blood [ms]
    """

    if model_id == 1:
        # Fit based on 1.5T, 3T, 7T data (Zhang et al.)
        t1 = 110 * b0 + 1316
        #t1 = np.around(t1) # TODO: think about if I want to remove the rounding. Philip had it.
        return t1
    elif model_id == 2:
        # (Rooney et al.)
        a = 3.35 # ms
        b = 0.340
        gamma = 42.58*1e6 # (gyromagnetic ratio)
        t1 = a*np.pow(gamma*b0, b)
        #t1 = np.around(t1) # TODO: think about if I want to remove the rounding. Philip had it.
        return t1
    elif model_id == 3:
        # Model between 1.5T and 7T.
        Hb = 5.15 # [mmol / l](mean corpuscular haemoglobin concentration)
        Y = 0.99 # (oyxgen saturation)
        Hct = 0.45 # (average normal Hematocrit)
        fe = (0.7 * Hct) / (0.7 * Hct + 0.95 * (1 - Hct)) # (water fraction in RBCs)
        R1p = fe * (1.099 - 0.057 * b0 + 0.033 * Hb * (1 - Y)) + (1 - fe) * (0.496 - 0.023 * b0)
        dT1 = 108 #[ms] (in vivo correction)
        t1 = 1000. / R1p + dT1
        #t1 = np.around(t1) # TODO: think about if I want to remove the rounding. Philip had it.
        return t1

def t2a_estimate(b0, model_id):
    """

    :param b0:
    :param model_id:
    :return:
    """
    if model_id == 1:
        # Value measured in Brooks et al. (10.1002/jmri.1880050414)
        return 1000 / 4.
    elif model_id == 2:
        # Model by Thomas et al. (10.1007/s10334-021-00993-2)
        # I actually can't replicate the b0 formula shown here by Philip. It looks more like arterial blood (about 100%
        # sO2) should have an R2 between 3.2 Hz and 4. Hz for 0.35T, as extrapolated from their Fig. 3.
        # Table 3 also suggests a T2 between 210 ms and 300 ms.
        Y = 0.99 # (oyxgen saturation)
        T20 = 270. # [ms](intrinsic T2 of blood)
        R2 = 1000. / T20 + (0.25 * np.pow(b0 / 0.2, 1.5)) * Y * Y # (approximated)
        T2 = 1000. / R2
        #T2 = np.around(T2) # TODO: think about if I want to remove the rounding. Philip had it.
        return T2

def calculate_fwhm(x_arr, y_arr):
    """
    Calculate FWHM very simply assuming no sidelobes are more than half the height of the maximum.
    :param x_arr: float array: x values
    :param y_arr: float array: y values, should be same shape as x_arr
    :return: tuple with three elements (float, int, int): First the full width half max value, then the two indices
    representing the edge of the function region that is above half max.
    """
    max_index = np.argmax(y_arr)
    max_val = y_arr[max_index]

    is_over_half_arr = y_arr > (max_val / 2.)

    last_index_below_half = np.nonzero(is_over_half_arr)[0][0] - 1 # Highest index that does not yet cross the threshold
    first_index_below_half = np.nonzero(is_over_half_arr)[0][-1] + 1 # Lowest index after peak that does cross the threshold

    fwhm = x_arr[first_index_below_half] - x_arr[last_index_below_half]

    return fwhm, last_index_below_half, first_index_below_half