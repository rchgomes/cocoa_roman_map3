import numpy as np

def f_h3(x1, x2, x3): #https://arxiv.org/pdf/astro-ph/0207454.pdf between eq 12 and 13
    return(np.sqrt(2*x1**2+2*x2**2-x3**2)/2)

def f_psi3(x1,x2,x3):
    h3 = f_h3(x1,x2,x3)
    phi3 = np.arccos((x3**2-x1**2-x2**2)/(-2*x1*x2))
    #print("phi3", phi3)
    sin_of_2_psi = (x2**2-x1**2)*x1*x2*np.sin(phi3)/(h3*x3)**2
    cos_of_2_psi = ((x2**2-x1**2)**2 - 4*(x1*x2)**2*np.sin(phi3)**2)/(4*(h3*x3)**2)
    #print("precos", (x2**2-x1**2)**2 - 4*(x1*x2)**2*np.sin(phi3)**2, 4*(h3*x3)**2)
    #print("intermediates", sin_of_2_psi, cos_of_2_psi)
    return(np.arctan2(sin_of_2_psi, cos_of_2_psi)/2)

def f_psi1(x1, x2, x3):
    return(f_psi3(x2, x3, x1))

def f_psi2(x1,x2,x3):
    return(f_psi3(x3,x1,x2))

def transform_gamma(gamma, num, r, u, v):

    x2 = r * np.pi / (60 * 180)
    x3 = u * x2
    x1 = v * x3 + x2

    psi1 = f_psi1(x1,x2,x3)
    psi2 = f_psi2(x1,x2,x3)
    psi3 = f_psi3(x1,x2,x3)

    #print("transforming", psi1, psi2, psi3)
    if num == 0:
        gamma_transf = gamma * np.exp(-2*1j*(psi1+psi2+psi3))

    if num == 1:
        gamma_transf = gamma * np.exp(-2 * 1j * (-psi1 + psi2 + psi3))
        #gamma_transf = gamma * np.exp(-2 * 1j * (psi1 + psi2 + psi3)) #manual adjust?

    if num ==2:
        gamma_transf = gamma * np.exp(-2 * 1j * (psi1 - psi2 + psi3))

    if num == 3:
        gamma_transf = gamma * np.exp(-2 * 1j * (psi1 + psi2 - psi3))

    return(gamma_transf)

def transform_gamma_factor(num, r, u, v):

    x2 = r * np.pi / (60 * 180)
    x3 = u * x2
    x1 = v * x3 + x2

    psi1 = f_psi1(x1,x2,x3)
    psi2 = f_psi2(x1,x2,x3)
    psi3 = f_psi3(x1,x2,x3)

    #print("transforming", psi1, psi2, psi3)
    if num == 0:
        #gamma_transf = np.exp(-2*1j*(psi1+psi2+psi3))
        gamma_transf = np.cos(-2*(psi1+psi2+psi3))
        print(psi1, psi2, psi3)
    if num == 1:
        gamma_transf = np.exp(-2 * 1j * (-psi1 + psi2 + psi3))
        #gamma_transf = gamma * np.exp(-2 * 1j * (psi1 + psi2 + psi3)) #manual adjust?

    if num ==2:
        gamma_transf = np.exp(-2 * 1j * (psi1 - psi2 + psi3))

    if num == 3:
        gamma_transf = np.exp(-2 * 1j * (psi1 + psi2 - psi3))

    return(np.real(gamma_transf))

def remove_zeros(x):

    x_loc = (x==0)
    x_val = 10**(-10)*x_loc
    return(x+x_val)

def uhat(x):

    return(x**2/2*np.exp(-x**2/2))