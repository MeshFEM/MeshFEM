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

def format(d, print_string = True):
    """
    Print or return a string representation of a benchmarking dictionary `d`.
    (for example, format(to_dict) should do the same thing as report()).
    """
    s = ''
    fullTime = 0
    for k in sorted(d.keys(), key=lambda x: (x.lower(), x)):
        t = d[k]
        if k == '':
            fullTime = t.time
            continue
        s += ((k.count(':') * 4) * ' ') + k.rpartition(':')[-1] + f'\t{t.time}\t{t.invocations}\n'
    s += f'Full time:\t{fullTime}'
    if (print_string): print(s)
    else: return s

################################################################################
# Functionality for accumulating multiple benchmark dictionaries into a single one.
################################################################################
from dataclasses import dataclass

@dataclass
class PyBenchmarkRecord:
    invocations: int = 0
    time: float = 0.0

    @property
    def averageTime(self) -> float:
        return self.time / self.invocations if self.invocations else 0.0

def sum(dicts):
    accumulated = {}

    for d in dicts:
        for key, record in d.items():
            if key not in accumulated:
                accumulated[key] = PyBenchmarkRecord()
            accumulated[key].invocations += record.invocations
            accumulated[key].time += record.time

    return accumulated

################################################################################
# Visualizations
################################################################################
def pieChart(heading='', d = None, includeOutside = False):
    """
    Create a pie chart visualizing the timing breakdown for a specified subtree
    of the benchmarking data.

    Specifically, given a subtree root, we plot the fraction of the subtree time
    spent within each of the subtrees rooted at the children.

    If `includeOutside` is true, we also plot the time spent outside the subtree
    for reference.
    """
    import numpy as np
    from matplotlib import pyplot as plt

    if d is None: d = to_dict()
    if (heading != ''): total_subtree_time = totalTime(heading + '$', d)
    else:               total_subtree_time = totalTime('^$', d)
    labels = []
    times = []
    if (heading != ''): subtimes = query(heading + ':[^:]+$', d)
    else:               subtimes = query('^[^:]+$', d)
    for k in subtimes:
        labels.append(k.split(':')[-1])
        times.append(d[k].time)

    full_time = totalTime('^$', d)
    unaccounted = total_subtree_time - np.sum(times)
    labels.append(f'Unaccounted for: {unaccounted / total_subtree_time:0.2%} ({unaccounted / full_time:0.2%} of full)')
    times.append(unaccounted)

    if includeOutside:
        labels.append('Outside')
        times.append(full_time - total_subtree_time)

    plt.title('Timing Breakdown' + (f' for {heading} ({total_subtree_time / full_time:0.2%} of Full)' if heading != '' else ''))

    wedges, _ = plt.pie(times, labels=None)

    # If we're including the "outside" label, we want to color it light gray.
    if includeOutside: wedges[-1].set_facecolor('lightgray')

    plt.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
