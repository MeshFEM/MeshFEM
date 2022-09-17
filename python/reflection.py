import inspect

def hasArg(func, argName):
    try:
        return argName in inspect.signature(func).parameters
    except:
        # Pybind11 methods/funcs apparently don't support `inspect.signature`,
        # but at least their arg names are guaranteed to appear in the docstring... :(
        return argName in func.__doc__

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
