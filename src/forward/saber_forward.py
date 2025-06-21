import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# Helper for extrapolation (like Hmisc::approxExtrap)
def approx_extrap(x, y, xout):
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
    return f(xout)

# Placeholder for snell_law function
def snell_law(view, sun):
    n_air = 1.0
    n_w = 1.33
    view_rad = np.deg2rad(view)
    sun_rad = np.deg2rad(sun)
    view_w = np.arcsin((n_air / n_w) * np.sin(view_rad))
    sun_w = np.arcsin((n_air / n_w) * np.sin(sun_rad))
    # Fresnel reflectance (simplified)
    rho_L = 0.021  # Placeholder, implement as needed
    return {'view_w': view_w, 'sun_w': sun_w, 'rho_L': rho_L}

def Saber_forward(
    chl=4.96, acdom440=0.9322314, anap440=0.07, bbp_550=0.00726002,
    z=2, rb_fraction=None, slope_parametric=False, plot=False, verbose=False,
    realdata=None, realdata_exist=True, wavelength=None,
    type_case_water=1, type_Rrs_below='deep', type_Rrs_water='below_surface',
    view=30, sun=30, P=1013.25, RH=0.5, alpha=1.0, Hoz=0.3, WV=2.0,
    f_dd=0.5, f_ds=0.5, g_dd=0.5, g_dsr=0.25, g_dsa=0.25
):
    # Set working directory if needed
    # os.chdir("/home/musk0001/R_inverse_wasi")

    # Read input spectra (example)
    demo_rrs = pd.read_csv("./data/input-spectra/demo_rrs_Om.csv")
    if wavelength is None:
        wavelength = demo_rrs['wavelength'].values
    if realdata is None:
        realdata = demo_rrs['rrs.demo'].values

    # --- Absorption ---
    abs_water = pd.read_csv("./data/input-spectra/abs_W.A", delim_whitespace=True, header=None, names=['wl', 'abs'])
    absorpt_W = approx_extrap(abs_water['wl'], abs_water['abs'], wavelength)

    # Plankton absorption
    phyto = pd.read_csv("./data/input-spectra/A0_A1_PhytoPlanc.dat", delim_whitespace=True, header=None, names=['wl', 'a0', 'a1'])
    a0 = approx_extrap(phyto['wl'], phyto['a0'], wavelength)
    a1 = approx_extrap(phyto['wl'], phyto['a1'], wavelength)
    aph_440 = 0.06 * (chl ** 0.65)
    abs_ph = (a0 + a1 * np.log(aph_440)) * aph_440
    abs_ph[abs_ph < 0] = 0

    # CDOM absorption
    S_CDOM = 0.014
    abs_CDOM_440 = acdom440
    abs_CDOM = abs_CDOM_440 * np.exp(-S_CDOM * (wavelength - 440))

    # NAP absorption
    S_X = 0.01160
    abs_X_440 = anap440
    abs_X = abs_X_440 * np.exp(-S_X * (wavelength - 440))

    # Plot absorption components
    if plot:
        plt.figure()
        plt.plot(wavelength, abs_ph, label='a_ph')
        plt.plot(wavelength, abs_CDOM, label='a_CDOM')
        plt.plot(wavelength, abs_X, label='a_NAP')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorption (1/m)')
        plt.legend()
        plt.show()

    # --- Scattering and backscattering ---
    if type_case_water == 1:
        b1 = 0.00144
    else:
        b1 = 0.00111
    lambda1 = 500
    bb_W = b1 * (wavelength / lambda1) ** -4.32

    refexponent = 0.46
    bb_x = bbp_550 * (wavelength / 550) ** -refexponent

    # --- Total IOPs ---
    abs_total = absorpt_W + abs_ph + abs_CDOM + abs_X
    bb_total = bb_W + bb_x
    ext = abs_total + bb_total
    omega_b = bb_total / ext

    # --- RT Model ---
    geometry_below = snell_law(view, sun)
    sun_w = geometry_below['sun_w']
    view_w = geometry_below['view_w']
    rho_L = geometry_below['rho_L']

    if type_case_water == 1:
        f_rs = 0.095
    else:
        f_rs = 0.0512 * (1 + (4.6659 * omega_b) + (-7.8387 * (omega_b ** 2)) + (5.4571 * (omega_b ** 3))) * (1 + (0.1098 / np.cos(sun_w))) * (1 + (0.4021 / np.cos(view_w)))

    Rrs_below_deep = f_rs * omega_b

    if type_Rrs_below == 'deep':
        Rrs_below = Rrs_below_deep
    else:
        # Implement shallow water logic here (bottom reflectance, etc.)
        Rrs_below = Rrs_below_deep  # Placeholder

    # --- Above surface reflectance ---
    # Implement atmospheric correction and skyglint as needed

    # --- Final Rrs computation ---
    Rrs = Rrs_below  # Placeholder for above-surface logic

    # --- Plot Rrs ---
    if plot:
        plt.figure()
        plt.plot(wavelength, Rrs, label='Rrs_model')
        if realdata_exist:
            plt.plot(wavelength, realdata, '--', label='Rrs_actual')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Rrs (1/sr)')
        plt.legend()
        plt.show()

    # --- Objective function for inverse mode ---
    if realdata_exist:
        Res = np.sum((realdata - Rrs) ** 2)
        Res_spectral = (Rrs - realdata) * 100 / realdata
        return {
            'data': pd.DataFrame({'wavelength': wavelength, 'Rrs': Rrs, 'p.bias': Res_spectral}),
            'ss.residual': Res,
            'method': 'SSR euclidean'
        }
    else:
        return {
            'data': pd.DataFrame({'wavelength': wavelength, 'Rrs': Rrs})
        }

# Example usage:
# result = Saber_forward(plot=True)