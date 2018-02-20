# Compute the relative error between the MeshFEM relative error and the Mathematica relative error.
import glob, subprocess as sp
def relError(a, b):
    num = abs(a - b)
    if (abs(b) == 0):
        if (abs(a) == 0):
            return 0
        return num / abs(a)
    return num / abs(b)
def relErrors():
    d1RelErrors = [];
    d2RelErrors = [];
    origin = [];
    for d in glob.glob('results/skip_*/poisson_*'):
        for comp in ['x', 'y']:
            for sample in range(3):
                quantity = "u_%s[%i]" % (comp, sample)
                meshFEM = sp.check_output(["python", "errorData.py", d, quantity])
                mathematica = sp.check_output(["python", "errorData.py", d, "mathematica " + quantity])
                for mfemLine, mathLine in zip(meshFEM.strip().split('\n'), mathematica.strip().split('\n')):
                    mfemD1, mfemD2 = map(float, mfemLine.split()[2:]);
                    mathD1, mathD2 = map(float, mathLine.split()[2:]);
                    d1RelErrors.append(relError(mfemD1, mathD1))
                    d2RelErrors.append(relError(mfemD2, mathD2))
                    origin.append("%s:%s:%i" % (d, comp, sample))
    return d1RelErrors, d2RelErrors, origin
