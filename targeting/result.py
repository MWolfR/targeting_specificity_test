import numpy
import pandas


target_classes_inh = ['SOM', 'AIS', 'SD', 'AD', 'PD', 'OTHER']
target_classes_exc = ['PD', 'AD', 'SD', 'OTHER']


class AxonTargets(object):
    """A Class to hold the results for axonal targeting, i.e. for a number of axons the types of postsynaptic
    structures they target with their synapses (one entry per synapse)"""

    def __init__(self, is_exc, res):
        """Inputs:
                is_exc: bool: Whether the dataset holds only excitatory axons.
                res: list of lists of strings: The raw data for axonal targeting. One list for each axon,
                in each list one string for each synapse. The string specifies the postsynaptic target class
                (one of SOM, AIS, SD, AD, PD, OTHER."""
        self.target_classes = target_classes_inh
        self.is_exc = is_exc
        if is_exc:
            self.target_classes = target_classes_exc
        if len(res) == 0:
            self.raw = []
            self.res = pandas.DataFrame([], columns=self.target_classes)
            return
        self.raw = [pandas.Categorical(_res, categories=self.target_classes)
                    for _res in res]
        self.res = pandas.concat([_raw.value_counts() for _raw in self.raw], axis=1).transpose()

    def fit(self, for_class, fit_func):
        """Get a probability to target one of the postsynaptic target classes using the specified fit function."""
        R = self.res[for_class]
        N = self.res.sum(axis=1)
        return fit_func(R, N)

    def fit_all(self, fit_func):
        """Get the probabilities to target each of the postsynaptic target classes using the specified fit function.
        Inputs:
            fit_func: Function to use for the fit. Can be found in targeting.specificity. Takes as input a numpy.array
            of dtype int specifying the total number of synapses for each axon and a numpy.array of dtype int specifying the
            number of synapses onto a given postsynaptic class for each axon."""
        p_vals = [self.fit(_cls, fit_func) for _cls in self.target_classes]
        return pandas.Series(p_vals, index=self.target_classes)

    def normalized(self):
        """Normalizes each row of the results and returns a copy with the normalized data."""
        res = AxonTargets(self.is_exc, [])
        res.raw = self.raw.copy()
        data = self.res.astype(float)
        data_sum = data.sum(axis=1)
        res.res = pandas.DataFrame(numpy.array([data[col] / data_sum
                                                for col in data.columns]).transpose(),
                                   columns=data.columns)
        return res

    def __len__(self):
        return len(self.raw)

    def __add__(self, other):
        if not isinstance(other, AxonTargets):
            raise Exception("Addition only valid for AxonTargets + AxonTargets. Got {t_class}".format(t_class=other.__class__))
        is_exc = numpy.all([self.is_exc, other.is_exc])
        A = self.res
        B = other.res
        if len(A.columns) > len(B.columns):
            B = B.reindex(A.columns, axis=1, fill_value=0)
        elif len(B.columns) > len(A.columns):
            A = A.reindex(B.columns, axis=1, fill_value=0)

        new_obj = AxonTargets(is_exc, [])
        new_obj.raw = self.raw + other.raw
        new_obj.res = pandas.concat([A, B], ignore_index=True)
        return new_obj

