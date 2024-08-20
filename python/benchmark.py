from _benchmark import *
import functools
import time

def benchmarkit(fn):
    # Ensure that the name and docstring of 'fn' is preserved in 'wrapper'
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      # the wrapper passes all parameters to the function being decorated
      start_timer_section(fn.__name__)
      res = fn(*args, **kwargs)
      stop_timer_section(fn.__name__)
      return res
    return wrapper

def benchmarkit_customname(name):
  def named_benchmarkit(fn):
      # Ensure that the name and docstring of 'fn' is preserved in 'wrapper'
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        # the wrapper passes all parameters to the function being decorated
        start_timer_section(name)
        res = fn(*args, **kwargs)
        stop_timer_section(name)
        return res
      return wrapper
  return named_benchmarkit

################################################################################
# Convenience routines for analyzing benchmark records returned by `to_dict`.
################################################################################
def query(pattern, d = None):
    """
    Return a dictionary holding all benchmark records with names matching `pattern`.
    """
    import re
    p = re.compile(pattern)
    if d is None: d = to_dict()
    result = {}
    for k in d:
        if (re.search(p, k)):
            result[k] = d[k]
    return result

def totalTime(pattern, d=None, default=None):
    """
    Total time across all benchmark records with names matching `pattern`.
    """
    d = query(pattern, d)
    if len(d) == 0:
        if default is not None: return default
        raise Exception(f'No records matching pattern {pattern}')
    result = 0
    for b in d.values():
        result += b.time
    return result

def totalTimePerInvocation(pattern, d=None, default=None):
    """
    Total time across all benchmark records with names matching `pattern`
    divided by the number of invocations of those code sections.
    Raises an exception if not all matching records have been invoked the same number of times.
    """
    d = query(pattern, d)
    if len(d) == 0:
        if default is not None: return default
        raise Exception(f'No records matching pattern {pattern}')
    total = 0
    invocations = []
    for b in d.values():
        total += b.time
        invocations.append(b.invocations)
    if (min(invocations) != max(invocations)): raise Exception("Inconsistent invocation count")
    return total / invocations[0]

def numInvocations(pattern, d=None, default=None):
    d = query(pattern, d)
    if len(d) == 0:
        if default is not None: return default
        raise Exception(f'No records matching pattern {pattern}')
    invocations = [b.invocations for b in d.values()]
    if (min(invocations) != max(invocations)): raise Exception("Inconsistent invocation count")
    return invocations[0]
