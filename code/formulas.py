import numpy as np

def cauchy(parameters, wavelength):
    A, B, C = parameters
    _array = A + B/wavelength**2 + C/wavelength**4
    return _array


def TL_e1(parameters, ph_energy):
    A, E0, C, Eg, einf = parameters
    alpha_cp = 4 * (E0**2) - (C**2)
    gamma_cp = (E0**2) - ((C ** 2) / 2)
    alpha = np.sqrt(alpha_cp)
    gamma = np.sqrt(gamma_cp)
    a_ln = ((Eg**2) - (E0**2)) * (ph_energy**2) + (Eg**2 * C**2) - (E0**2 * (E0**2 + 3*Eg**2))
    a_tan = ((ph_energy**2) - (E0**2)) * ((E0**2) + (Eg**2)) + ((Eg**2) * (C**2))
    dzeta4 = ((ph_energy**2) - (gamma**2))**2 + (((alpha**2) * (C**2))/4)

    e1 = einf + \
    (((A*C)/(np.pi*dzeta4)) * (a_ln/(2*alpha*E0)) * np.log(((E0**2) + (Eg**2) + (alpha*Eg))/((E0**2) + (Eg**2) - (alpha*Eg)))) \
    - ((A/(np.pi*dzeta4)) * (a_tan/E0)*(np.pi - np.arctan((2*Eg+alpha)/C) + np.arctan(((-2*Eg) + alpha)/C))) \
    + (2*((A*E0)/(np.pi*dzeta4*alpha)) * Eg*((ph_energy**2) - (gamma**2))*(np.pi + 2*np.arctan(2*((gamma**2) - (Eg**2))/(alpha*C)))) \
    - (((A*E0*C)/(np.pi*dzeta4)) * (((ph_energy**2) + (Eg**2))/ph_energy) * np.log(abs(ph_energy-Eg)/(ph_energy+Eg)))\
    + ((2*A*E0*C)/(np.pi*dzeta4)) *Eg * np.log((abs(ph_energy-Eg)*(ph_energy+Eg))/(np.sqrt(((E0**2) - (Eg**2))**2 + ((Eg**2) * (C**2)))))
    e1 = np.nan_to_num(e1)
    return np.where((alpha_cp <= 0 or gamma_cp <= 0 or E0 <= 0 or C <= 0 or ((E0**2 + Eg**2 - alpha*Eg ) < 0)), 0, e1)

def TL_e2(parameters, ph_energy):
    A, E0, C, Eg = parameters
    e2 = ((A*E0*C*((ph_energy-Eg)**2)) / (((ph_energy**2) - (E0**2))**2 + ((C**2)*ph_energy**2))) * (1/ph_energy)
    return np.where(ph_energy > Eg, e2, 0)

def TL(parameters, ph_energy):
    '''
        A

        E0

        C

        Eg

        einf
    '''
    return np.stack([TL_e1(parameters, ph_energy), TL_e2(parameters[:-1], ph_energy)], axis=1)



    
