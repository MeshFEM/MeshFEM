import inspect

def hasArg(func, argName):
    try:
        return argName in inspect.signature(func).parameters
    except:
        # Pybind11 methods/funcs apparently don't support `inspect.signature`,
        # but at least their arg names are guaranteed to appear in the docstring... :(
        return argName in func.__doc__

def hasMethod(obj, methodName):
    try: return callable(obj.__getattribute__(methodName))
    except: return False

def evalWithCustomArgs(f, customArgs, mandatoryArgs=[], strict=False):
    '''
    Evaluate f(*mandatoryArgs) also passing the additional args or kwargs in `customArgs`.
    Any `kwargs` that are not expected by `f` are simply filtered out if `strict == False`
    or reported with an exception if `strict == True`.
    '''
    if (customArgs is not None):
        if (isinstance(customArgs, list)): return f(*mandatoryArgs, *customArgs)
        if (isinstance(customArgs, dict)):
            if strict:
                missing = [k for k in customArgs.keys() if not hasArg(f, k)]
                if len(missing) > 0: raise Exception('Missing keys: [' + ', '.join(missing) + ']')
            return f(*mandatoryArgs, **{k: v for k, v in customArgs.items() if hasArg(f, k)})
        return f(*mandatoryArgs, customArgs)
    return f(*mandatoryArgs)

def baseClassNames(obj):
    """
    Get the names of all of the base classes of an object instance.
    """
    return [b.__name__ for b in type.mro(obj.__classes__)]

def isElasticSolid(obj):
    return 'ElasticSolid' in obj.__class__.__name__

def isVoxelFEMSimulator(obj):
    return 'TensorProductSimulator' in obj.__class__.__name__

def isNumeric(obj):
    import numbers
    return isinstance(obj, numbers.Number)
