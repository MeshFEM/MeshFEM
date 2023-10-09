import numpy as np
from numpy.linalg import norm
import sparse_matrices
from reflection import hasArg, evalWithCustomArgs, hasMethod

def genPerturbation(x):
    return np.random.uniform(low=-1,high=1, size=x.shape)

def preamble(obj, xeval, perturb, customArgs, fixedVars = []):
    if (xeval   is None): xeval = evalWithCustomArgs(obj.getVars, customArgs)
    if (perturb is None): perturb = genPerturbation(xeval)
    xold = evalWithCustomArgs(obj.getVars, customArgs)
    perturb = np.copy(perturb)
    perturb[fixedVars] = 0.0
    return (xold, xeval, perturb)

def setVars(obj, x, customArgs = None):
    if (customArgs is not None):
        obj.setVars(x, **{k: v for k, v in customArgs.items() if hasArg(obj.setVars, k)})
    else:
        obj.setVars(x)

def basisDirection(obj, c):
    e_c = np.zeros(obj.numVars())
    e_c[c] = 1.0
    return e_c

def fdDelta(obj, fd_eps, xeval = None, perturb = None, customArgs = None, fixedVars = []):
    xold, xeval, perturb = preamble(obj, xeval, perturb, customArgs, fixedVars)

    if (hasMethod(obj, 'energy')):
        def evalAt(x):
            setVars(obj, x, customArgs)
            return evalWithCustomArgs(obj.energy, customArgs)
    elif (hasMethod(obj, 'value')):
        def evalAt(x):
            setVars(obj, x, customArgs)
            return evalWithCustomArgs(obj.value, customArgs)
    else: raise Exception('Object does not implement `energy` or `value`')

    fd_delta_E = (evalAt(xeval + perturb * fd_eps) - evalAt(xeval - perturb * fd_eps)) / (2 * fd_eps)
    setVars(obj, xold, customArgs)

    return fd_delta_E

def fdGrad(obj, fd_eps, xeval = None, customArgs = None, fixedVars = []):
    """
    Construct a finite difference approximation of the full gradient by perturbing one variable at a time.
    """
    return np.array([fdDelta(obj, fd_eps, perturb=basisDirection(obj, i), xeval=xeval, customArgs=customArgs, fixedVars=fixedVars) for i in range(obj.numVars())])

def fdHessian(obj, fd_eps, xeval = None, customArgs = None, fixedVars = []):
    """
    Construct a finite difference approximation of the full Hessian by perturbing one variable at a time.
    """
    nv = obj.numVars()
    fd_hess = np.zeros((nv, nv))
    if (xeval is None): xeval = obj.getVars()
    x = xeval.copy()
    for i in range(nv):
        d = basisDirection(obj, i)
        obj.setVars(x + fd_eps * d)
        g_plus = obj.gradient()
        obj.setVars(x - fd_eps * d)
        g_minus = obj.gradient()
        obj.setVars(x)
        fd_hess[:, i] = (g_plus - g_minus) / (2 * fd_eps)
    return fd_hess

def validateGrad(obj, fd_eps = 1e-6, xeval = None, perturb = None, customArgs = None, fixedVars = [], g = None):
    """
    Return (fd, an) where `fd` and `an` are the finite difference approximation and
    analytically calculated gradient.
    """
    xold, xeval, perturb = preamble(obj, xeval, perturb, customArgs, fixedVars)

    setVars(obj, xeval, customArgs)
    if g is None: g = evalWithCustomArgs(obj.gradient, customArgs)
    if (g.ravel().shape == perturb.ravel().shape): # Use `ravel()` to support scalar-valued functions of matrices
        analytic_delta_E = g.ravel().dot(perturb.ravel())
    analytic_delta_E = g.dot(perturb)              # Don't use `ravel()` in the case of vector-valued functions where "grad" is really a Jacobian...

    fd_delta_E = fdDelta(obj, fd_eps, xeval, perturb, customArgs, fixedVars)

    return (fd_delta_E, analytic_delta_E)

def findBadGradComponent(obj, fd_eps, xeval = None, customArgs = None, fixedVars = [], nprobes = 3):
    """
    Use a simple binary search to hone in on bad components of the gradient.
    This isn't guaranteed to find the worst component, but it should find one
    of the worse ones.
    """
    xold, xeval, perturb = preamble(obj, xeval, None, customArgs, fixedVars)

    setVars(obj, xeval, customArgs)
    g = evalWithCustomArgs(obj.gradient, customArgs)

    # Determine the total error across `nprobes` perturbations of interval
    # the [lowIdx, upIdx] (inclusive)
    def errForRange(lowIdx, upIdx):
        if lowIdx > upIdx: return 0
        err = 0
        for i in range(nprobes):
            perturb = np.random.uniform(low=-1, high=1, size=evalWithCustomArgs(obj.numVars, customArgs))
            perturb[fixedVars] = 0
            perturb[0:lowIdx] = 0
            perturb[upIdx + 1:] = 0

            fd_delta_E = fdDelta(obj, fd_eps, xeval, perturb, customArgs, fixedVars)
            analytic_delta_E = g.dot(perturb)
            err += np.abs(analytic_delta_E - fd_delta_E) / np.abs(fd_delta_E)
        return err

    lowIdx, upIdx = 0, len(perturb)
    while upIdx > lowIdx:
        mid = (lowIdx + upIdx) // 2
        error_low = errForRange(lowIdx, mid)
        error_up  = errForRange(mid + 1, upIdx)
        # print([lowIdx, mid, upIdx, error_low, error_up])
        if error_low > error_up:
            upIdx = mid
        else:
            lowIdx = mid + 1

    setVars(obj, xold, customArgs)

    return lowIdx

def validateHessian(obj, fd_eps = 1e-6, xeval = None, perturb = None, customArgs = None, fixedVars = [], indexInterval = None, H = None, testHessVec = False, hessVecPrecomp = None):
    """
    Validate a Hessian (testHessVec = False) or Hessian-vector product (testHessVec = True).
    Returns
    -------
        relative error (in l2 norm)
        finite difference delta gradient
        analytic delta gradient
    """
    xold, xeval, perturb = preamble(obj, xeval, perturb, customArgs, fixedVars)

    def gradAt(x):
        setVars(obj, x, customArgs)
        return evalWithCustomArgs(obj.gradient, customArgs)

    setVars(obj, xeval, customArgs)
    if testHessVec:
        if H is not None: raise Exception('Hessian should not be passed if testHessVec is True!')
        if hessVecPrecomp is None:
            an_delta_grad = evalWithCustomArgs(obj.hessVec, customArgs, mandatoryArgs=[perturb])
        else: an_delta_grad = hessVecPrecomp
    else:
        if H is None: H = evalWithCustomArgs(obj.hessian, customArgs)
        if isinstance(H, np.ndarray): # Dense case
            an_delta_grad = H @ perturb
        else: an_delta_grad = H.apply(perturb)

    fd_delta_grad = (gradAt(xeval + perturb * fd_eps) - gradAt(xeval - perturb * fd_eps)) / (2 * fd_eps)
    if indexInterval is not None:
        fd_delta_grad = fd_delta_grad[indexInterval[0]:indexInterval[1]]
        an_delta_grad = an_delta_grad[indexInterval[0]:indexInterval[1]]

    setVars(obj, xold, customArgs)

    return (norm(an_delta_grad - fd_delta_grad) / norm(fd_delta_grad), fd_delta_grad, an_delta_grad)

def gradConvergence(obj, perturb=None, customArgs=None, fixedVars = [], epsilons=None):
    if epsilons is None:
        epsilons = np.logspace(-9, -3, 100)
    errors = []
    if (perturb is None): perturb = preamble(obj, None, perturb, customArgs, fixedVars)[-1]
    g = evalWithCustomArgs(obj.gradient, customArgs)
    for eps in epsilons:
        fd, an = validateGrad(obj, g=g, customArgs=customArgs, perturb=perturb, fd_eps=eps, fixedVars=fixedVars)
        err = np.linalg.norm(an - fd) / np.linalg.norm(an)
        # print(f'{err=}')
        errors.append(err)
    return (epsilons, errors, an)

from matplotlib import pyplot as plt
def gradConvergencePlotRaw(obj, perturb=None, customArgs=None, fixedVars = [], epsilons=None):
    eps, errors, ignore = gradConvergence(obj, perturb, customArgs, fixedVars, epsilons=epsilons)
    plt.loglog(eps, errors, label='grad')
    plt.grid()

def gradConvergencePlot(obj, perturb=None, customArgs=None, fixedVars = [], epsilons=None):
    gradConvergencePlotRaw(obj, perturb, customArgs, fixedVars, epsilons=epsilons)
    title = 'Directional derivative fd test for gradient'
    if hasattr(obj, 'name'): title += ' - ' + obj.name()
    plt.title(title)
    plt.ylabel('Relative error')
    plt.xlabel('Step size')

def hessConvergence(obj, perturb=None, customArgs=None, fixedVars = [], indexInterval = None, epsilons=None, testHessVec=False):
    if epsilons is None:
        epsilons = np.logspace(-9, -3, 100)
    errors = []
    if (perturb is None): perturb = np.random.uniform(-1, 1, size=evalWithCustomArgs(obj.numVars, customArgs))

    H = None
    hessVecPrecomp = None
    if testHessVec:
        hessVecPrecomp = evalWithCustomArgs(obj.hessVec, customArgs, mandatoryArgs=[perturb])
    else:
        if H is None: H = evalWithCustomArgs(obj.hessian, customArgs)
        H = evalWithCustomArgs(obj.hessian, customArgs)
    for eps in epsilons:
        err, fd, an = validateHessian(obj, customArgs=customArgs, perturb=perturb, fd_eps=eps, fixedVars=fixedVars, indexInterval=indexInterval, H=H, testHessVec=testHessVec, hessVecPrecomp=hessVecPrecomp)
        errors.append(err)
    return (epsilons, errors, an)

def hessConvergencePlotRaw(obj, perturb=None, customArgs=None, fixedVars = [], indexInterval = None, epsilons=None, testHessVec=False):
    eps, errors, ignore = hessConvergence(obj, perturb, customArgs, fixedVars, indexInterval=indexInterval, epsilons=epsilons, testHessVec=testHessVec)
    plt.loglog(eps, errors, label='hess')
    plt.grid()

def hessConvergencePlot(obj, perturb=None, customArgs=None, fixedVars = [], indexInterval = None, epsilons=None, testHessVec=False):
    hessConvergencePlotRaw(obj, perturb, customArgs, fixedVars, indexInterval=indexInterval, epsilons=epsilons, testHessVec=testHessVec)
    title = 'Directional derivative fd test for Hessian'
    if hasattr(obj, 'name'): title += ' - ' + obj.name()
    plt.title(title)
    plt.ylabel('Relative error')
    plt.xlabel('Step size')

def block_hessian_error(obj, var_indices, va, vb, customArgs=None, eps=1e-6, perturb=None, testHessVec = False, hessVecPrecomp = None):
    '''
    Report the error in the (va, vb) block of the Hessian, where
    va and vb are members of the `var_types` array.
    '''
    if (perturb is None):
        perturb = np.random.normal(0, 1, evalWithCustomArgs(obj.numVars, customArgs))
    block_perturb = np.zeros_like(perturb)
    block_perturb[var_indices[vb]] = perturb[var_indices[vb]]
    err, fd_delta_grad, an_delta_grad = validateHessian(obj, customArgs=customArgs, fd_eps=eps, perturb=block_perturb, testHessVec=testHessVec, hessVecPrecomp=hessVecPrecomp)
    fd_delta_grad = fd_delta_grad[var_indices[va]]
    an_delta_grad = an_delta_grad[var_indices[va]]
    return (norm(an_delta_grad - fd_delta_grad) / norm(an_delta_grad),
            fd_delta_grad, an_delta_grad)

def hessian_convergence_block_plot(obj, var_types, var_indices, customArgs=None, testHessVec = False, hessVecPrecomp = None, plot_name=None):
    perturb = np.random.normal(0, 1, evalWithCustomArgs(obj.numVars, customArgs))
    numVarTypes = len(var_types)
    epsilons = np.logspace(np.log10(1e-12), np.log10(1e2), 50)
    fig = plt.figure(figsize=(16, 12))
    for i, vi in enumerate(var_types):
        for j, vj in enumerate(var_types):
            plt.subplot(numVarTypes, numVarTypes, i * numVarTypes + j + 1)
            errors = [block_hessian_error(obj, var_indices, vi, vj, customArgs=customArgs, eps=eps, perturb=perturb, testHessVec=testHessVec, hessVecPrecomp=hessVecPrecomp)[0] for eps in epsilons]
            plt.loglog(epsilons, errors)
            plt.title(f'({vi}, {vj}) block')
            plt.grid()
            plt.tight_layout()
    if (plot_name is not None):
        plt.savefig(plot_name, dpi = 300)
    plt.show()

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



def fdSecondDerivative(obj, fd_eps, xeval = None, customArgs = None, fixedVars = []):
    xold, xeval, perturb = preamble(obj, xeval, None, customArgs, fixedVars)

    if (hasMethod(obj, 'energy')):
        def evalAt(x):
            setVars(obj, x, customArgs)
            return evalWithCustomArgs(obj.energy, customArgs)
    elif (hasMethod(obj, 'value')):
        def evalAt(x):
            setVars(obj, x, customArgs)
            return evalWithCustomArgs(obj.value, customArgs)
    else: raise Exception('Object does not implement `energy` or `value`')

    fd_delta_E = (evalAt(xeval + fd_eps) + evalAt(xeval - fd_eps) - 2 * evalAt(xeval)) / (fd_eps ** 2)
    setVars(obj, xold, customArgs)
    return fd_delta_E

def validateSecondDerivative(obj, fd_eps = 1e-6, xeval = None, customArgs = None, fixedVars = [], second_derivative = None):
    xold, xeval, perturb = preamble(obj, xeval, None, customArgs, fixedVars)
    setVars(obj, xeval, customArgs)
    if (second_derivative is None): second_derivative = evalWithCustomArgs(obj.secondDerivative, customArgs)
    fd_delta_E = fdSecondDerivative(obj, fd_eps, xeval, customArgs, fixedVars)

    return (fd_delta_E, second_derivative)

def secondDerivativeConvergence(obj, customArgs=None, fixedVars = [], epsilons=None):
    if epsilons is None:
        epsilons = np.logspace(-9, -3, 100)
    errors = []
    second_derivative = evalWithCustomArgs(obj.secondDerivative, customArgs)
    for eps in epsilons:
        fd, an = validateSecondDerivative(obj, second_derivative=second_derivative, customArgs=customArgs, fd_eps=eps, fixedVars=fixedVars)
        err = np.linalg.norm(an - fd) / np.linalg.norm(an)
        # print(f'{err=}')
        errors.append(err)
    return (epsilons, errors, an)


from matplotlib import pyplot as plt
def secondDerivativeConvergencePlotRaw(obj, customArgs=None, fixedVars = [], epsilons=None):
    eps, errors, ignore = secondDerivativeConvergence(obj, customArgs, fixedVars, epsilons=epsilons)
    plt.loglog(eps, errors, label='grad')
    plt.grid()

def secondDerivativeConvergencePlot(obj, customArgs=None, fixedVars = [], epsilons=None):
    secondDerivativeConvergencePlotRaw(obj, customArgs, fixedVars, epsilons=epsilons)
    title = 'Directional derivative fd test for second derivative'
    if hasattr(obj, 'name'): title += ' - ' + obj.name()
    plt.title(title)
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
