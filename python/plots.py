import numpy as np
from matplotlib import pyplot as plt

def tensionForcePolarPlot(strainMag, psis, psiLabels, validateUnaxial = None):
    """
    Plots the magnitude of tension required to produce a uniaxial strain state
    with magntiude "strainMag" as a function of the strain direction.
    Note, due to Poisson's ratio effects additional compressive forces may also
    be needed.
    """
    F = np.array([[1 + strainMag, 0], [0, 1 - strainMag]])
    R = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    if not isinstance(psis, list):
        psis = [psis]
        psiLabels = [psiLabels]
    if validateUnaxial is None:
        validateUnaxial = len(psis) * [False]
    forces = [[] for i in range(len(psis))]

    thetas = np.linspace(0, 2 * np.pi, 1000)
    for th in thetas:
        Frot = F @ R(th).T
        for i, psi in enumerate(psis):
            if hasattr(psi, 'setDeformationGradient'):
                Fpad = np.zeros_like(psi.getDeformationGradient())
                Fpad[0:2, 0:2] = Frot
                psi.setDeformationGradient(Fpad)
            elif hasattr(psi, 'setC'): psi.setC(Frot.T @ Frot)
            else: raise Exception('Unrecognized interface.')
            l = np.linalg.eigh(psi.PK2Stress())[0]
            if (validateUnaxial[i]):
                assert(np.abs(l[0]) < 1e-5 * np.abs(l[1]))
            forces[i].append(l[1])
    ax = plt.subplot(111, projection='polar')
    for i, l in enumerate(psiLabels):
        plt.plot(thetas, forces[i], label=l + f' (anistropy: {np.max(forces[i]) / np.min(forces[i]):.2f})')
    plt.tick_params(axis='y', )
    ax.set_ylim([0, 1.05 * np.max(forces)])
    ax.get_yaxis().set_major_formatter(plt.NullFormatter())
    plt.legend()
    plt.tight_layout()
