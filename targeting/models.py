from .result import AxonTargets
from scipy import stats
import numpy


hs_dend_to_type = {8: 'SOM', 9: 'PD', 5: 'SD', 1: 'AD', 2: 'AIS', 4: 'OTHER'}
hs_type_to_dend = dict([(v, k) for k, v in hs_dend_to_type.items()])
hs_type_to_axon = {'corticocortical': 1, 'thalamocortical': 2,
                   'inhibitory': 3, 'other': 4}
hs_type_to_exc = {'corticocortical': True, 'thalamocortical': True,
                   'inhibitory': False, 'other': False}


class DegreesFromData(object):
    """Helper class to generate out-degrees for control models,
    i.e. the distribution of the number of synapses on an axon fragment.
    Reproduces the exact distribution of a reference dataset."""
    def __init__(self, results):
        """Input: A reference dataset of type AxonTargets."""
        self.samples = results.res.sum(axis=1).values

    def rvs(self, n):
        """Generate random variates.
        Input: n, the number of variates to return. Will be cast to int
        Output: numpy.array, shape (n,) of random variates."""
        return numpy.random.choice(self.samples, int(n))


def binomial_model(reference_model, fit_func, oversample=1.0,
                   oversample_individual_axons=1.0):
    """Generates random results of axonal targeting for a binomial control model.
    Inputs:
        reference_model, class: targeting.AxonTargets: the reference model from which we
        estimate out-degree distributions and probabilities to target the individual postsynaptic structure classes.
        fit_func: a function object that is used to estimate the binomial probabilities. Use one from targeting.specificity
        or write your own!
        oversample: float: factor by which to over (or under-) sample the data, i.e. for generating data for more (or fewer)
        axons than in the reference dataset.
        oversample_individual_axons: float: Similarly, this one oversamples the outdegrees, i.e. generates more targets
        for each individual axon."""
    deg_obj = DegreesFromData(reference_model)
    N = deg_obj.rvs(len(reference_model) * oversample)
    N = (oversample_individual_axons * N).astype(int)
    p = reference_model.fit_all(fit_func)
    if p.sum() != 1.0:
        print("""Warning: The model fit is inconsistent!
        Total probabilities add up to {ttl_p}.
        Adjusting probability values for 'OTHER' to compensate!""".format(ttl_p=p.sum()))
        if 'OTHER' in p and (p.sum() - p['OTHER']) <= 1.0:
            p['OTHER'] = 1.0 - p.sum() + p['OTHER']
        elif p.sum() < 1.0:
            p['OTHER'] = 1.0 - p.sum()
        else:
            raise Exception("Adjustment not possible!")
    cum_p = numpy.hstack([0, numpy.cumsum(p.values)])

    def sample(cum_p, labels, n):
        r = numpy.random.rand(n)
        res = [numpy.sum((r >= mn_v) & (r < mx_v))
               for mn_v, mx_v in zip(cum_p[:-1], cum_p[1:])]
        out = []
        for lbl, r in zip(labels, res):
            out.extend(r * [lbl])
        return out
    result = [sample(cum_p, p.index, n) for n in N]
    return AxonTargets(reference_model.is_exc, result)


def subsample_from_appositions_model(reference_model, fit_fun, filling_fraction,
                                     n_sampled_together, oversample=1.0):
    """Generates random results of axonal targeting for a two-stage model. First, it uses a binomial model with
    increased out-degrees (i.e. higher number of synapses per axon) to generate synapse 'candidates'. Then, it groups
    candidates that are targeting the same type of postsynaptic structure into groups of a specified size. Finally, each
    group of candidates is either accepted or rejected with a fixed, independent probability. Accepted groups lead to the
    formation of a synapse in this model. The acceptance probability is balanced with the increase of out-degrees
    in the initial binomial model, such that the final out-degrees match the reference model.

        Inputs:
            reference_model, class: targeting.AxonTargets: the reference model from which we
            estimate out-degree distributions and probabilities to target the individual postsynaptic structure classes.
            fit_func: a function object that is used to estimate the binomial probabilities. Use one from targeting.specificity
            or write your own!
            filling_fraction: float: The probability to 'accept' a group of candidate synapses. The amount of oversampling in
            the initial binomial model is the inverse of this.
            n_sampled_together: int: The size of a group of candidate synapses that are accepted or rejected together.
            That means N candidate synapses will be grouped into ceil(N / n_sampled_together) groups. If set to 1, this
            is identical to a fully binomial model!
            oversample: float: factor by which to over (or under-) sample the data, i.e. for generating data for more (or fewer)
            axons than in the reference dataset.
            """
    apposition_mdl = binomial_model(reference_model, fit_fun, oversample=oversample,
                                    oversample_individual_axons=(1.0/filling_fraction))

    def sample_single_col(N, col_label):
        grouped = numpy.arange(0, N + n_sampled_together, n_sampled_together)
        grouped[-1] = N
        success = numpy.random.rand(len(grouped) - 1) <= filling_fraction
        return numpy.diff(grouped)[success].sum() * [col_label]
    res = []
    import progressbar
    pbar = progressbar.ProgressBar()
    for i in pbar(apposition_mdl.res.index):
        row = apposition_mdl.res.loc[i]
        out_row = []
        for col_lbl, N in row.iteritems():
            out_row.extend(sample_single_col(N, col_lbl))
        res.append(out_row)
    return AxonTargets(reference_model.is_exc, res)


def Helmstaedter_results(filenames, axon_classes,
                         min_n_dendrite=10, min_n_axon=10,
                         filter_unknown_dend=False, raw=False):
    """Reads and returns the data of the Science paper as an AxonTargets object.
        Inputs: filenames: dict: filenames[synapses] is the path to the synapses.hdf5 file,
                                 filenames[axons] is the path to the axons.hdf5 file,
                                 filenames[dendrites] is the path to the dendrites.hdf5 file.
                Get these files from: https://l4dense2019.brain.mpg.de/
                axon_classes: list of strings: The types of axons to return the data for. A list of one or several of
                the following: 'thalamocortical', 'corticocortical', 'inhibitory', 'other'.
                min_n_dendrite=10: Dendrite fragments with fewer than that number of synapses are ignored.
                min_n_axon=10: Axon fragments with fewer than that number of synapses are ignored.
                filter_unknown_dend=False: If set to True, dendrite fragments of unknown type (around 50%) are filtered out.
                raw=False: If set to True, returns a list of lists of postsynaptic target types instead."""
    import h5py
    h5_syns = h5py.File(filenames['synapses'], 'r')
    h5_axons = h5py.File(filenames['axons'], 'r')
    h5_dends = h5py.File(filenames['dendrites'], 'r')
    src_classes = [hs_type_to_axon[axon_class] for axon_class in axon_classes]

    is_exc = [hs_type_to_exc[axon_class] for axon_class in axon_classes]
    is_exc = numpy.all(is_exc)

    def filter_by_min_count(indices, min_n, also_remove_zero=True):
        bins = numpy.arange(numpy.max(indices) + 2)
        idv_counts = numpy.histogram(indices, bins=bins)[0]
        valid = numpy.nonzero(idv_counts >= min_n)[0]
        if also_remove_zero:
            valid = valid[valid > 0]
        return numpy.in1d(indices, valid)

    try:
        axons = numpy.array(h5_syns['synapses']['preAxonId'])
        dendrites = numpy.array(h5_syns['synapses']['postDendriteId'])
        axon_classes = numpy.array(h5_axons['axons']['class'])
        dend_classes = numpy.array(h5_dends['dendrites']['class'])

        """Apply filtering by minimal count. This also removes not-reconstructed axons and dendrites."""
        valid = filter_by_min_count(axons, min_n_axon) & filter_by_min_count(dendrites, min_n_dendrite)
        axons = axons[valid]
        dendrites = dendrites[valid]

        """Get the types of pre- and postsynaptic structures. Minus one because dendrites=0 (or axons=0) indicates a
        not-reconstructed neurite. We filtered these out in the previous step."""
        syn_targets = dend_classes[dendrites - 1]
        syn_sources = axon_classes[axons - 1]
        """Check which ones are from one of the specified presynaptic classes."""
        valid = numpy.in1d(syn_sources, src_classes)
        """And if desired, also check if the postsynaptic class is 'OTHER'"""
        if filter_unknown_dend:
            valid = valid & (syn_targets != hs_type_to_dend['OTHER'])
        axons = axons[valid]
        dendrites = dendrites[valid]
        syn_sources = syn_sources[valid]
        syn_targets = syn_targets[valid]

        """Now assemble the structure."""
        per_axon = {}
        for tgt, ax in zip(syn_targets, axons):
            per_axon.setdefault(ax, []).append(hs_dend_to_type[tgt])

        if raw:
            return list(per_axon.values())
        return AxonTargets(is_exc, list(per_axon.values()))
    except:
        raise
    finally:
        h5_syns.close()
        h5_axons.close()
        h5_dends.close()

