# targeting_specificity_test
Evaluating the connectome analyses of a Science paper of the Helmstaedter group

In this jupyter notebook, I explore some aspects of the analysis performed in figure 4 of the following paper:
Dense connectomic reconstruction in layer 4 of the somatosensory cortex (Science, 2019; DOI: 10.1126/science.aay3134)

The authors reconstructed connectivity in a neocortical volume with EM methods. They then analyzed the results with respect to the question to what degree the connectivity is targeted. For all axons reconstructed in their volume, they consider the type of postsynaptic structures innervated: Smooth dendrites, proximal dendrites (i.e. any dendrite where the soma is also in the reconstructed volume), apical dendrites, somata and axon initial segments. They then try to analyze which axons exhibit a preference for one or several of these types of structure.

However, I believe their analysis is flawed and their conclusions are invalid. Their results are not indication of targeting preference, but only prove that synapse formation is not statistically independent between synapses.

In this notebook, I first recreate their analysis from the description in their materials and methods and the indivial panels of their figure. I then run a number of sanity checks, such as re-running the same analysis on a simple control model that should yield negative results to validate my code further. Finally, I define another control mode and re-run the analysis on it, finding even stronger results than the original data. That control, however, is a simple stochastic model with no targeting specificity at all. Thereby, I prove that the interpretation of their results is too strong: They can only conclude that synapse formation is not statistically independent, i.e. the selection of postsynaptic structures is not a binomial process, but that does not necessarily indicate a prescribed targeting preference.

To follow my analysis, simply run the included jupyter notebook:

git clone git@github.com:MWolfR/targeting_specificity_test.git

cd targeting_specificity_test

jupyter notebook Targeting specificity test.ipynb

Required python packages are:
pandas
numpy
scipy
targeting (included in this repository)
