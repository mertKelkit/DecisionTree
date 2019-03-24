import numpy as np

import scipy.stats.distributions as dist

from scipy import stats
from collections import Counter
from scipy.stats import chi2_contingency


def calculate_chi_square(y_parts, targets, alpha=0.05, is_frequency=False):
    if is_frequency:
        frequencies = y_parts
    else:
        f = lambda part: frequency(part, targets)
        frequencies = np.array(list(map(f, y_parts))).transpose()

    _, _, df, expected_f = chi2_contingency(frequencies)

    chi2_statistic = np.sum(stats.chisquare(frequencies, expected_f).statistic)

    p_val = dist.chi2.sf(chi2_statistic, df)

    significant = p_val <= alpha

    return chi2_statistic, significant


def frequency(arr, targets=None):
    counter = Counter(arr)

    if targets is None:
        targets = list(set(arr))

    freq = np.array([counter[k] for k in targets])

    return freq


def one_sided_entropy(y, targets):
    return stats.entropy(frequency(y, targets))
