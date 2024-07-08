import numpy as np
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import differential_evolution

def eV_to_um(eV):
    return 1.2398/eV

def um_to_eV(um):
    return 1.2398/um

def multi_model(parameters, *args):
    '''
    args:
        x_space
        
        functions
        
        param_sizes
    '''
    x_space = args[0] #ph_energy or wavelength
    functions = args[1]
    param_sizes = args[2]

    y_pred = np.sum([functions[i](parameters[sum(param_sizes[:i]): sum(param_sizes[:i+1])], x_space) for i in range(len(functions))], axis=0)

    return y_pred

def fitness_func(parameters, *args):
    '''
    args:
        x_space

        y_true
        
        functions
        
        param_sizes
    '''
    x_space = args[0] #ph_energy or wavelength
    y_true = args[1]
    functions = args[2]
    param_sizes = args[3]
    if len(functions) > 1:
        y_pred = multi_model(parameters, x_space, functions, param_sizes)
        #np.sum([functions[i](parameters[sum(param_sizes[:i]): sum(param_sizes[:i+1])], x_space) for i in range(len(functions))], axis=0)
    else:
        y_pred = functions[0](parameters, x_space)
        
    #loss = np.where(np.isnan(y_pred).any(), 5, root_mean_squared_error(y_true, y_pred))
    if np.isnan(y_pred).any():
        return 100
    else:
        loss = root_mean_squared_error(y_true, y_pred)
        return loss
    
def fit_OC(x, y, functions, functions_parameters_number, bounds):
    finder = differential_evolution(
        func=fitness_func,
        bounds= bounds,
        args=(x, y, functions, functions_parameters_number),
        init='sobol',
        polish=True
    )
    return finder.x, finder.fun


def theta_j(N_i, N_j, theta_i):
    return np.arcsin((N_i*np.sin(theta_i))/N_j)

def beta(thickness, wavelength, N, theta):
    return (2*np.pi*thickness)/wavelength * N * np.cos(theta)

def N(n,k):
    return n-1j*k

def e_to_n(e1, e2):
    return np.sqrt((e1 + np.sqrt(e1**2 + e2**2))/2)

def e_to_k(e1, e2):
    return np.sqrt((-e1 + np.sqrt(e1**2 + e2**2))/2)

def rs(N_i=None, N_j=None, theta_i=None, theta_j=None, Ss=None):
    if Ss is not None:
        #print(Ss)
        return Ss[1,0]/\
               Ss[0,0]
    else:
        return ((N_i * np.cos(theta_i))-(N_j * np.cos(theta_j)))/\
               ((N_i * np.cos(theta_i))+(N_j * np.cos(theta_j)))

def rp(N_i=None, N_j=None, theta_i=None, theta_j=None, Sp=None):
    if Sp is not None:
        return Sp[1,0]/Sp[0,0]
    else:
        return ((N_j * np.cos(theta_i))-(N_i * np.cos(theta_j)))/\
               ((N_j * np.cos(theta_i))+(N_i * np.cos(theta_j)))
    
def t_s(N_i, N_j, theta_i, theta_j):
    return (2*N_i*np.cos(theta_i))/\
        (N_i*np.cos(theta_i)+N_j*np.cos(theta_j))

def t_p(N_i, N_j, theta_i, theta_j):
    return (2*N_i*np.cos(theta_i))/\
        (N_j*np.cos(theta_i)+N_i*np.cos(theta_j))

def M_n(beta, r, t):
    return np.array([[np.exp(-1j*beta), 0], [0, np.exp(1j*beta)]]) @ np.array([[1, r], [r, 1]]) / t

def M(r01, t01, M_ns):
    M = (np.array([[1, r01], [r01, 1]]) / t01)
    for M_n in M_ns:
        M = M @ M_n
    return M

def psi(r_p, r_s):
    return np.arctan(np.abs(r_p/r_s))

def delta(r_p, r_s):
    angle = np.angle(r_p/r_s)
    return np.where(angle < 0, angle + 2*np.pi, angle)