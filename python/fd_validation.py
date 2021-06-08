import numpy as np
from numpy.linalg import norm
import sparse_matrices
from reflection import hasArg

def genPerturbation(x):
    return np.random.uniform(low=-1,high=1, size=x.shape)

def preamble(obj, xeval, perturb, fixedVars = []):
    if (xeval   is None): xeval = obj.getVars()
    if (perturb is None): perturb = genPerturbation(xeval)
    xold = obj.getVars()
    perturb = np.copy(perturb)
    perturb[fixedVars] = 0.0
    return (xold, xeval, perturb)

def setVars(obj, x, customArgs = None):
    if (customArgs is not None):
        obj.setVars(x, **{k: v for k, v in customArgs.items() if hasArg(obj.setVars, k)})
    obj.setVars(x)

def evalWithCustomArgs(f, customArgs):
    if (customArgs is not None):
        if (isinstance(customArgs, list)): return f(*customArgs)
        if (isinstance(customArgs, dict)): return f(**{k: v for k, v in customArgs.items() if hasArg(f, k)})
        return f(customArgs)
    return f()

def basisDirection(obj, c):
    e_c = np.zeros(obj.numVars())
    e_c[c] = 1.0
    return e_c

def fdGrad(obj, fd_eps, xeval = None, perturb = None, customArgs = None, fixedVars = []):
    xold, xeval, perturb = preamble(obj, xeval, perturb, fixedVars)

    def evalAt(x):
        setVars(obj, x, customArgs)
        return evalWithCustomArgs(obj.energy, customArgs)

    fd_delta_E = (evalAt(xeval + perturb * fd_eps) - evalAt(xeval - perturb * fd_eps)) / (2 * fd_eps)
    setVars(obj, xold, customArgs)

    return fd_delta_E

def validateGrad(obj, fd_eps = 1e-6, xeval = None, perturb = None, customArgs = None, fixedVars = [], g = None):
    xold, xeval, perturb = preamble(obj, xeval, perturb, fixedVars)

    setVars(obj, xeval, customArgs)
    if g is None: g = evalWithCustomArgs(obj.gradient, customArgs)
    analytic_delta_E = g.dot(perturb)

    fd_delta_E = fdGrad(obj, fd_eps, xeval, perturb, customArgs, fixedVars)
    setVars(obj, xold, customArgs)

    return (fd_delta_E, analytic_delta_E)

def findBadGradComponent(obj, fd_eps, xeval = None, customArgs = None, fixedVars = [], nprobes = 3):
    """
    Use a simple binary search to hone in on bad components of the gradient.
    This isn't guaranteed to find the worst component, but it should find one
    of the worse ones.
    """
    xold, xeval, perturb = preamble(obj, xeval, None, fixedVars)

    setVars(obj, xeval, customArgs)
    g = evalWithCustomArgs(obj.gradient, customArgs)

    # Determine the total error across `nprobes` perturbations of interval
    # the [lowIdx, upIdx] (inclusive)
    def errForRange(lowIdx, upIdx):
        if lowIdx > upIdx: return 0
        err = 0
        for i in range(nprobes):
            perturb = np.random.uniform(low=-1, high=1, size=obj.numVars())
            perturb[fixedVars] = 0
            perturb[0:lowIdx] = 0
            perturb[upIdx + 1:] = 0

            fd_delta_E = fdGrad(obj, fd_eps, xeval, perturb, customArgs, fixedVars)
            analytic_delta_E = g.dot(perturb)
            err += np.abs(analytic_delta_E - fd_delta_E)
        return err

    lowIdx, upIdx = 0, len(perturb)
    while upIdx > lowIdx:
        # print([lowIdx, upIdx])
        mid = (lowIdx + upIdx) // 2
        if errForRange(lowIdx, mid) > errForRange(mid + 1, upIdx):
            upIdx = mid
        else:
            lowIdx = mid + 1

    setVars(obj, xold, customArgs)

    return lowIdx

def validateHessian(obj, fd_eps = 1e-6, xeval = None, perturb = None, customArgs = None, fixedVars = [], indexInterval = None, H = None):
    """
    Returns
    -------
        relative error (in l2 norm)
        finite difference delta gradient
        analytic delta gradient
    """
    xold, xeval, perturb = preamble(obj, xeval, perturb, fixedVars)

    def gradAt(x):
        setVars(obj, x, customArgs)
        return evalWithCustomArgs(obj.gradient, customArgs)

    setVars(obj, xeval, customArgs)
    if H is None: H = evalWithCustomArgs(obj.hessian, customArgs)
    fd_delta_grad = (gradAt(xeval + perturb * fd_eps) - gradAt(xeval - perturb * fd_eps)) / (2 * fd_eps)
    if isinstance(H, np.ndarray): # Dense case
        an_delta_grad = H @ perturb
    else: an_delta_grad = H.apply(perturb)

    if indexInterval is not None:
        fd_delta_grad = fd_delta_grad[indexInterval[0]:indexInterval[1]]
        an_delta_grad = an_delta_grad[indexInterval[0]:indexInterval[1]]

    setVars(obj, xold, customArgs)

    return (norm(an_delta_grad - fd_delta_grad) / norm(fd_delta_grad), fd_delta_grad, an_delta_grad)

def gradConvergence(obj, perturb=None, customArgs=None, fixedVars = [], epsilons=None):
    if epsilons is None:
        epsilons = np.logspace(-9, -3, 100)
    errors = []
    if (perturb is None): perturb = np.random.uniform(-1, 1, size=obj.numVars())
    g = evalWithCustomArgs(obj.gradient, customArgs)
    for eps in epsilons:
        fd, an = validateGrad(obj, g=g, customArgs=customArgs, perturb=perturb, fd_eps=eps, fixedVars=fixedVars)
        err = np.abs(an - fd) / np.abs(an)
        errors.append(err)
    return (epsilons, errors, an)

from matplotlib import pyplot as plt
def gradConvergencePlotRaw(obj, perturb=None, customArgs=None, fixedVars = [], epsilons=None):
    eps, errors, ignore = gradConvergence(obj, perturb, customArgs, fixedVars, epsilons=epsilons)
    plt.loglog(eps, errors, label='grad')
    plt.grid()

def gradConvergencePlot(obj, perturb=None, customArgs=None, fixedVars = [], epsilons=None):
    gradConvergencePlotRaw(obj, perturb, customArgs, fixedVars, epsilons=epsilons)
    plt.title('Directional derivative fd test for gradient')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')

def hessConvergence(obj, perturb=None, customArgs=None, fixedVars = [], indexInterval = None, epsilons=None):
    if epsilons is None:
        epsilons = np.logspace(-9, -3, 100)
    errors = []
    if (perturb is None): perturb = np.random.uniform(-1, 1, size=obj.numVars())

    H = evalWithCustomArgs(obj.hessian, customArgs)
    for eps in epsilons:
        err, fd, an = validateHessian(obj, customArgs=customArgs, perturb=perturb, fd_eps=eps, fixedVars=fixedVars, indexInterval=indexInterval, H=H)
        errors.append(err)
    return (epsilons, errors, an)

def hessConvergencePlotRaw(obj, perturb=None, customArgs=None, fixedVars = [], indexInterval = None, epsilons=None):
    eps, errors, ignore = hessConvergence(obj, perturb, customArgs, fixedVars, indexInterval=indexInterval, epsilons=epsilons)
    plt.loglog(eps, errors, label='hess')
    plt.grid()

def hessConvergencePlot(obj, perturb=None, customArgs=None, fixedVars = [], indexInterval = None, epsilons=None):
    hessConvergencePlotRaw(obj, perturb, customArgs, fixedVars, indexInterval=indexInterval, epsilons=epsilons)
    plt.title('Directional derivative fd test for Hessian')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')

def allEnergies(obj):
    if hasattr(obj, 'EnergyType'):
        return {name: obj.energy(etype) for name, etype in obj.EnergyType.__members__.items()}
    else:
        return {'Energy': obj.energy()}

def linesearchValidationPlot(obj, direction, alphaMax = 1e-5, width=12, height=6):
    """
    Help diagnose issues with the backtracking linesearch by plotting the
    energy along the linesearch direction `direction`.
    """
    x = obj.getVars()
    alphas = np.linspace(0, 1e-5, 100)
    energies = []
    for alpha in alphas:
        obj.setVars(x + alpha * direction)
        energies.append(allEnergies(obj))
    obj.setVars(x)
    keys = list(energies[0].keys())
    nplots = len(keys)
    plt.figure(figsize=(width, height))
    for i, k in enumerate(keys):
        cols = int(np.ceil(np.sqrt(nplots)))
        rows = int(np.ceil(nplots / cols))
        plt.subplot(rows, cols, i + 1)
        if k is None: plt.plot(alphas, energies)
        else: plt.plot(alphas, [e[k] for e in energies])
        if k is not None: plt.title(k)
        plt.grid()
    plt.tight_layout()
