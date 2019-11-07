import numpy
from scipy import stats


def neg_log_likelihood(px, R, N):
    """Calculates the negative log likelihood to find given results R under the 'first hit' binomial model
    with parameters px and N.
    Inputs:
       px (float): The probability to use in the binomial model
       R (np.array, shape: (l,), dtype: bool): The results in terms of how many of the n outcomes was positive.
       N (np.array, dtype: int): The array of values of n to use in the binomial model"""
    # Only first hit matters in this model: cast to bool
    R = R.astype(bool)
    # Generate vector of the probabilities to observe each individual result.
    # First step: Probability to observe a negative result, i.e. outcome of binomial is zero
    p_zero = (1 - px[0]) ** N
    # Second step: Where the outcome was positive we want the probability of that outcome instead
    # conveniently there are only two outcomes, so we just one minus it.
    p_zero[R == 1] = 1 - p_zero[R == 1]
    # Return negative log likelihood
    return -numpy.sum(numpy.log(p_zero))


def neg_log_likelihood_from_vec(px, res):
    """Calculates the negative log likelihood to find given results res under a binomial model
        with probability px.
        Inputs:
           px (float): The probability to use in the binomial model
           res (list of np.arrays): List of results. Each individual result is an array of dtype bool
           with one entry per synapse that details whether or not the outcome for that synapse was positive.
           """
    # Calculate whether the number of positive outcomes
    R = numpy.array(map(numpy.sum, res))
    # Get lengths
    N = numpy.array(map(len, res))
    return neg_log_likelihood(px, R, N)


def first_hit_fit(R, N):
    """Find the most likely value for p under the 'first hit' binomial model
        Inputs:
          R (np.array, shape: (l,), dtype: bool): The results in terms of how many of the n outcomes was positive.
          N (np.array, dtype: int): The array of values of n to use in the binomial model"""
    from scipy.optimize import minimize
    initial_guess = numpy.array([0.25])
    result = minimize(neg_log_likelihood, initial_guess, (R, N), bounds=[(1E-12, 1-1E-12)])
    assert result.success
    return result.x[0]


def first_hit_fit_from_vec(res):
    """Find the most likely value for p under the 'first hit' binomial model
            Inputs:
              res (list of np.arrays): List of results. Each individual result is an array of dtype bool
               with one entry per synapse that details whether or not the outcome for that synapse was positive.
              """
    R = numpy.array(map(numpy.sum, res))
    N = numpy.array(map(len, res))
    return first_hit_fit(R, N)


def trivial_fit(R, N):
    """Find the most likely value for p under the trivial binomial model
            Inputs:
              R (np.array, shape: (l,), dtype: bool): The results in terms of how many of the n outcomes was positive.
              N (np.array, dtype: int): The array of values of n to use in the binomial model"""
    total_positive = numpy.sum(R)
    total_count = numpy.sum(N)
    return float(total_positive) / float(total_count)


def trivial_fit_from_vec(res):
    """Find the most likely value for p under a trivial binomial model
       Inputs:
         res (list of np.arrays): List of results. Each individual result is an array of dtype bool
         with one entry per synapse that details whether or not the outcome for that synapse was positive."""
    return numpy.mean(numpy.hstack(res))


def create_expected_binomial(p, N, bins):
    """Create the distribution of the fraction of synapses onto a given type under the binomial model.
       Inputs:
         p (float): The p of the binomial probability distribution
         N (np.array, dtype: int): The array of values of n to use in the binomial model.
         bins (np.array): The bins to use to turn the results into a histogram."""
    raw = [stats.binom(n, p).pmf(range(n + 1))
           for n in N]
    ret = []
    for n, data in zip(N, raw):
        x = numpy.arange(len(data), dtype=float) / len(data)
        idxx = numpy.digitize(x, bins=bins)
        ret.append([data[idxx == i].sum() for i in range(1, len(bins))])
    return numpy.vstack(ret).sum(axis=0)


def significant_targeting_fraction(data, control, target_rate, expected_p=None):
    """Calculate the fraction of axons for which we can assume significant targeting when compared to the control.
       Inputs:
          data: class: AxonTargets: The data or modeled data to test
          control: class: AxonTargets: The control model to compare against
          target_rate: Target false detection rate to use
          expected_p (optional): The binomial probabilities to compare to in the test. If not specified, they
          are calculated from 'control'.
    """
    if expected_p is None:
        expected_p = control.fit_all(trivial_fit)
    """Look, there is also a simple analytical solution to this. But I am tired, so I'll just use the samples in control.
    TODO: Update later"""
    N = data.res.sum(axis=1)
    N_ctrl = control.res.sum(axis=1)

    keys = [_k for _k in data.res.columns if _k != 'OTHER']
    p = numpy.array([1.0 - stats.binom(N, expected_p[col]).cdf(data.res[col] - 1)
                     for col in keys])
    p_ctrl = numpy.array([1.0 - stats.binom(N_ctrl, expected_p[col]).cdf(control.res[col] - 1)
                          for col in keys])

    def false_detection_rate_criterion(p1, p2, target_rate):
        thresholds = numpy.hstack([0, numpy.unique(numpy.hstack([p1, p2]))])
        pos1 = numpy.array([(p1 < thresh).mean() for thresh in thresholds]).astype(float)
        pos2 = numpy.array([(p2 < thresh).mean() for thresh in thresholds]).astype(float)
        fdr = pos2 / (pos1 + 1E-12)
        if not numpy.any(fdr > target_rate):
            return thresholds[-1]
        idxx = numpy.nonzero(fdr > target_rate)[0][0] - 1
        #Note: idxx cannot be 0
        return thresholds[idxx]

    thresholds = numpy.array([false_detection_rate_criterion(_p, _ctrl, target_rate)
                              for _p, _ctrl in zip(p, p_ctrl)]).reshape((len(keys), 1))
    significants = p < thresholds
    # The following could be used to ensure that any axon is only targeting a single type.
    #for idx in numpy.nonzero(significants.sum(axis=0) > 1)[0]:
    #    significants[:, idx] = significants[:, idx] & (p[:, idx] == p[significants[:, idx], idx].min())
    res = dict([(k, (1.0 - target_rate) * val)
                for k, val in zip(keys, significants.mean(axis=1))])
    return res
