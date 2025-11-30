import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

""" Créé par Jonathan Cauchon pour le cours GEL-4203

Historique des modifs:

    - 20 sept 2020: guide_1d_analytique
        dans l'utilisation de la fonction estimate_beta, si aucune solution n'est trouvée, 
        guide_id_analytique augmente la resolution sur beta jusqu'à ce qu'une solution
        soit trouvée.



"""


def estimate_beta(wvl, t, n1, n2, n3, beta_res):
    """
        Function qui estime les solutions du guide d'onde plan
    """

    wvl = np.asarray(wvl)
    k = 2*np.pi/wvl
    beta = np.linspace(np.max([n1,n3])*k, n2*k, beta_res)[:-1]
    beta = beta if beta.ndim == 2 else np.expand_dims(beta, -1)
    wvl = wvl if wvl.ndim == 1 else np.expand_dims(wvl, 0)
    
    # TE
    h = np.sqrt((k*n2)**2 - beta**2)

    p = np.sqrt(beta**2 - (k*n3)**2)
    q = np.sqrt(beta**2 - (k*n1)**2)
    zero_TE = np.tan(h*t) - (p + q)/h/(1 - p*q/h**2)
    
    # TM
    p_ = p*(n2/n3)**2
    q_ = q*(n2/n1)**2
    zero_TM = np.tan(h*t) - (p_ + q_)/h/(1 - p_*q_/h**2)

    beta_TE = []
    beta_TM = []
    
    for i in range(beta.shape[-1]):
        # solution esitmée: croisement du zéro en direction négative (+ -> -)
        solutions_TE = np.where(np.sign(zero_TE[:-1,i]) - np.sign(zero_TE[1:,i]) == 2)[0]
        solutions_TM = np.where(np.sign(zero_TM[:-1,i]) - np.sign(zero_TM[1:,i]) == 2)[0]
        # print(solutions_TE, solutions_TM)
        # plt.plot(zero_TE)
        # plt.plot([0,beta.shape[0]], [0,0])
        # plt.ylim((-1,1))
        # plt.show()

        beta_sol_TE = beta[solutions_TE,i]
        beta_sol_TM = beta[solutions_TM,i]
        beta_TE.append(beta_sol_TE)
        beta_TM.append(beta_sol_TM)

    return beta_TE, beta_TM


def guide_1d_analytique(wvl, t, n1, n2, n3):
    """ Solution analytique du guide d'onde plan
        
    Paramètres:    
        wvl: longueur d'onde [m] (float, liste ou numpy array)
        t  : épaisseur du coeur [m] (float)
        n1, n2, n3: indices des trois régions (floats)
        
    retourne:
        neff_TE, neff_TM: les indices effectifs des modes TE et TM (les solutions).
                        Ce sont des listes de listes. Chaque sous-liste de la liste est pour
                        une longueur d'onde de 'wvl' et chaque item de la sous-liste contient
                        tous les indices supportés par le guide, en ordre croissant.
                        
    exemple d'usage:
        
        wvl = np.linspace(1530e-9,1560e-9,1000) # longueurs d'onde considérées
        neff_TE, neff_TM = guide_1d_analytique(wvl, 220e-9, 1.44, 3.45, 1.44)
        
        print(neff_TM[0][0]) # le mode TM fondamental (TM_0) à 1530 nm
    """

    wvl = np.asarray(wvl)
    wvl = wvl if wvl.ndim == 1 else np.expand_dims(wvl, 0)

    def equation_TE(beta):
        k = 2*np.pi/wvl
        h = np.sqrt((k*n2)**2 - beta**2)
        p = np.sqrt(beta**2 - (k*n3)**2)
        q = np.sqrt(beta**2 - (k*n1)**2)

        return np.tan(h*t) - (p + q)/h/(1 - p*q/h**2)
    
    def equation_TM(beta):
        k = 2*np.pi/wvl
        h = np.sqrt((k*n2)**2 - beta**2)
        p = np.sqrt(beta**2 - (k*n3)**2)
        q = np.sqrt(beta**2 - (k*n1)**2)
        p_ = p*(n2/n3)**2
        q_ = q*(n2/n1)**2

        return np.tan(h*t) - (p_ + q_)/h/(1 - p_*q_/h**2)

    beta_res = 1000
    beta_TE_estimate, beta_TM_estimate = estimate_beta(wvl, t, n1, n2, n3, beta_res)
    #print(beta_TE_estimate, beta_TM_estimate)  
    
    # si la solution est trop tight faut augmenter la résolution
    while (len(beta_TE_estimate[0]) < 1) or (len(beta_TM_estimate[0]) < 1):
        beta_res *= 10
        beta_TE_estimate, beta_TM_estimate = estimate_beta(wvl, t, n1, n2, n3, beta_res)

    neff_TE, neff_TM = [], []
    
    for i in range(wvl.shape[0]):
        for j in range(len(beta_TE_estimate[i])):
            beta_TE_estimate[i][j] = scipy.optimize.fsolve(equation_TE, beta_TE_estimate[i][j]).squeeze()
        for j in range(len(beta_TM_estimate[i])):
            beta_TM_estimate[i][j] = scipy.optimize.fsolve(equation_TM, beta_TM_estimate[i][j]).squeeze()

        neff_TE.append(np.flip(beta_TE_estimate[i]*wvl[i]/2/np.pi).tolist())
        neff_TM.append(np.flip(beta_TM_estimate[i]*wvl[i]/2/np.pi).tolist())
            

    return neff_TE, neff_TM



def tracer_modes(wvl, t, n1, n2, n3, plot_TE=True, plot_TM=True,
                 num_mode = [0]):
    # num_mode : ordre du mode à tracer - 1= mode fondamental
    mu0 = 4*np.pi*1e-7
    epsilon0 = 8.85e-12
    eta = np.sqrt(mu0/epsilon0)
    c = 3e8

    assert type(wvl) == float, "Utiliser une seule longueur d'onde"
    neff_TE, neff_TM = guide_1d_analytique(wvl, t, n1, n2, n3)

    numPoints = 1000
    x1 = np.linspace(-3*t, -t/2, numPoints)
    x2 = np.linspace(-t/2, t/2, numPoints)
    x3 = np.linspace( t/2, 3*t, numPoints)
    x = np.hstack((x1,x2, x3))
    nx = np.hstack((n1*np.ones(numPoints), n2*np.ones(numPoints), n3*np.ones(numPoints)))

    fig, ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].plot([-t/2*1e6,-t/2*1e6,t/2*1e6,t/2*1e6],[-2,2,2,-2], "k--")
    ax[1].plot([-t/2*1e6,-t/2*1e6,t/2*1e6,t/2*1e6],[-2,2,2,-2], "k--")
    
    
    if plot_TE:
        # for i, n in enumerate(neff_TE[0]):
        for ii in range(0,len(num_mode)): # pour pouvoir tracer seulement un mode voulu
            n = neff_TE[0][num_mode[ii]]
            beta = 2*np.pi*n/wvl
            k = 2*np.pi/wvl
            h = np.sqrt((k*n2)**2 - beta**2)
            p = np.sqrt(beta**2 - (k*n3)**2)
            q = np.sqrt(beta**2 - (k*n1)**2)
            E_field = np.hstack(( np.exp(q*(x1 + t/2)), 
                                (np.cos(h*(x2+t/2))+q/h*np.sin(h*(x2+t/2))),
                                (np.cos(h*t) + q/h*np.sin(h*t))*np.exp(-p*(x3 - t/2)) ))
            H_field = E_field*nx/eta
            ax[0].plot(x*1e6, E_field/np.max(np.abs(E_field)), label="TE"+str(num_mode[ii]))
            ax[1].plot(x*1e6, H_field/np.max(np.abs(H_field)), label="TE"+str(num_mode[ii]))

    if plot_TM:
        # for i, n in enumerate(neff_TM[0]):
        for ii in range(0,len(num_mode)): # pour pouvoir tracer seulement un mode voulu
            n = neff_TM[0][num_mode[ii]]
            beta = 2*np.pi*n/wvl
            k = 2*np.pi/wvl
            h = np.sqrt((k*n2)**2 - beta**2)
            p = np.sqrt(beta**2 - (k*n3)**2)
            q = np.sqrt(beta**2 - (k*n1)**2)
            p_ = p*(n2/n3)**2
            q_ = q*(n2/n1)**2
            H_field = np.hstack(( h/q_*np.exp(q*(x1 + t/2)), 
                                (h/q_*np.cos(h*(x2+t/2))+np.sin(h*(x2+t/2))),
                                (h/q_*np.cos(h*t) + np.sin(h*t))*np.exp(-p*(x3 - t/2)) ))
            H_field = np.hstack(( h/q_*np.exp(q*(x1+t/2)), 
                (h/q_*np.cos(h*(x2+t/2))+np.sin(h*(x2+t/2))), 
                (h/q_*np.cos(h*t)+np.sin(h*t))*np.exp(-p*(x3-t/2)) ))

            ax[1].plot(x*1e6, H_field/np.max(np.abs(H_field)), label="TM"+str(num_mode[ii]))  
            E_field = H_field*eta/nx  
            ax[0].plot(x*1e6, E_field/np.max(np.abs(E_field)), label="TM"+str(num_mode[ii]))

    
    ax[0].set_ylim((-1.1,1.1))
    ax[0].set_ylabel("E")
    ax[0].set_xlabel(r"y ($\mu$m)")
    ax[0].legend()

    
    ax[1].set_ylim((-1.1,1.1))
    ax[1].set_ylabel("H")
    ax[1].set_xlabel(r"y ($\mu$m)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()










