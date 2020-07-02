import numpy as np
from numpy.linalg import norm
import sparse_matrices

def preamble(obj, xeval, perturb, fixedVars = []):
    if (xeval   is None): xeval = obj.getVars()
    if (perturb is None): perturb = np.random.uniform(low=-1,high=1, size=obj.numVars())
    xold = obj.getVars()
    perturb = np.copy(perturb)
    perturb[fixedVars] = 0.0
    return (xold, xeval, perturb)

def evalWithCustomArgs(f, customArgs):
    if (customArgs is not None):
        if (isinstance(customArgs, list)): return f(*customArgs)
        return f(customArgs)
    return f()

def fdGrad(obj, fd_eps, xeval = None, perturb = None, customArgs = None, fixedVars = []):
    xold, xeval, perturb = preamble(obj, xeval, perturb, fixedVars)

    def evalAt(x):
        obj.setVars(x)
        return evalWithCustomArgs(obj.energy, customArgs)

    fd_delta_E = (evalAt(xeval + perturb * fd_eps) - evalAt(xeval - perturb * fd_eps)) / (2 * fd_eps)
    obj.setVars(xold)

    return fd_delta_E

def validateGrad(obj, fd_eps = 1e-6, xeval = None, perturb = None, customArgs = None, fixedVars = []):
    xold, xeval, perturb = preamble(obj, xeval, perturb, fixedVars)

    obj.setVars(xeval)
    g = evalWithCustomArgs(obj.gradient, customArgs)
    analytic_delta_E = g.dot(perturb)

    fd_delta_E = fdGrad(obj, fd_eps, xeval, perturb, customArgs, fixedVars)

    return (fd_delta_E, analytic_delta_E)

def validateHessian(obj, fd_eps = 1e-6, xeval = None, perturb = None, customArgs = None, fixedVars = []):
    xold, xeval, perturb = preamble(obj, xeval, perturb, fixedVars)

    def gradAt(x):
        obj.setVars(x)
        return evalWithCustomArgs(obj.gradient, customArgs)

    obj.setVars(xeval)
    h = evalWithCustomArgs(obj.hessian, customArgs)
    fd_delta_grad = (gradAt(xeval + perturb * fd_eps) - gradAt(xeval - perturb * fd_eps)) / (2 * fd_eps)
    if isinstance(h, np.ndarray): # Dense case
        analytic_delta_grad = h @ perturb
    else: analytic_delta_grad = h.apply(perturb)

    obj.setVars(xold)

    return (norm(analytic_delta_grad - fd_delta_grad) / norm(fd_delta_grad), fd_delta_grad, analytic_delta_grad)

def gradConvergence(obj, perturb=None, customArgs=None, fixedVars = []):
    epsilons = np.logspace(-9, -3, 100)
    errors = []
    if (perturb is None): perturb = np.random.uniform(-1, 1, size=obj.numVars())
    for eps in epsilons:
        fd, an = validateGrad(obj, customArgs=customArgs, perturb=perturb, fd_eps=eps, fixedVars = fixedVars)
        err = np.abs(an - fd) / np.abs(an)
        errors.append(err)
    return (epsilons, errors, an)

from matplotlib import pyplot as plt
def gradConvergencePlotRaw(obj, perturb=None, customArgs=None, fixedVars = []):
    eps, errors, ignore = gradConvergence(obj, perturb, customArgs, fixedVars)
    plt.loglog(eps, errors, label='grad')
    plt.grid()

def gradConvergencePlot(obj, perturb=None, customArgs=None, fixedVars = []):
    gradConvergencePlotRaw(obj, perturb, customArgs, fixedVars)
    plt.title('Directional derivative fd test for gradient')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')

def hessConvergence(obj, perturb=None, customArgs=None, fixedVars = []):
    epsilons = np.logspace(-9, -3, 100)
    errors = []
    if (perturb is None): perturb = np.random.uniform(-1, 1, size=obj.numVars())
    for eps in epsilons:
        err, fd, an = validateHessian(obj, customArgs=customArgs, perturb=perturb, fd_eps=eps, fixedVars = fixedVars)
        errors.append(err)
    return (epsilons, errors, an)

def hessConvergencePlotRaw(obj, perturb=None, customArgs=None, fixedVars = []):
    eps, errors, ignore = hessConvergence(obj, perturb, customArgs, fixedVars)
    plt.loglog(eps, errors, label='hess')
    plt.grid()

def hessConvergencePlot(obj, perturb=None, customArgs=None, fixedVars = []):
    hessConvergencePlotRaw(obj, perturb, customArgs, fixedVars)
    plt.title('Directional derivative fd test for Hessian')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
