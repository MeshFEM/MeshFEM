import inspect

def hasArg(func, argName):
    try:
        return argName in inspect.signature(func).parameters
    except:
        # Pybind11 methods/funcs apparently don't support `inspect.signature`,
        # but at least their arg names are guaranteed to appear in the docstring... :(
        return argName in func.__doc__
