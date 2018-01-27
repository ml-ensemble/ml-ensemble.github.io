# -*- coding: utf-8 -*-
"""

.. _learner_tutorial:


.. currentmodule:: mlens.parallel

Learner Mechanics
=================

ML-Ensemble is designed to provide an easy user interface. But it is also designed
to be extremely flexible, all the wile providing maximum concurrency at minimal
memory consumption. The lower-level API that builds the ensemble and manages the
computations is constructed in as modular a fashion as possible.

The low-level API introduces a computational graph-like environment that you can
directly exploit to gain further control over your ensemble. In fact, building
your ensemble through the low-level API is almost as straight forward as using the
high-level API. In this tutorial, we will walk through the basics :class:`Learner`
and :class:`Transformer` class.


The Learner API
^^^^^^^^^^^^^^^

Basics
------

The base estimator of ML-Ensemble is the :class:`Learner` instance. A learner is a
wrapper around a generic estimator along with a cross-validation strategy. The job
of the learner is to manage all sub-computations required for fitting and prediction.
In fact, it's public methods are generators from sub-learners, that do the actual
computation.  A learner is the parent node of an estimator's computational sub-graph
induced by the cross-validation strategy.

A learner is created by specifying an ``estimator`` and an ``indexer``, along with a
set of optional arguments, most notably the ``name`` of the learner. Naming is important,
is it is used for cache referencing. If setting it manually, ensure you give the learner
a unique name.
"""
from mlens.utils.dummy import OLS
from mlens.parallel import Learner, Job
from mlens.index import FoldIndex


indexer = FoldIndex(folds=2)
learner = Learner(estimator=OLS(),
                  indexer=indexer,
                  name='ols')

######################################################################
# The learner doesn't do any heavy lifting itself, it manages the creation a sub-graph
# of auxiliary :class:`SubLearner` nodes for each fold during estimation.
# This process is dynamic: the sub-learners are temporary instances created for each
# estimation.

######################################################################
# To fit a learner, we need a cache reference. When fitting all estimators from the
# main process, this reference can be a list. If not (e.g. multiprocessing), the
# reference should instead be a ``str`` pointing to the path of the cache directory.
# Prior to running a job (``fit``, ``predict``, ``transform``), the learner must be
# configured on the given data by calling the ``setup`` method. This takes cares of
# indexing the training set for cross-validation, assigning output columns et.c.
import os, tempfile
import numpy as np

X = np.arange(20).reshape(10, 2)
y = np.random.rand(10)

# Specify a cache directory
path = []

# Run the setup routine
learner.setup(X, y, 'fit')

# Run
for sub_learner in learner.gen_fit(X, y):
    sub_learner.fit(path)

print("Cached items:\n%r" % path)

############################################################################
# Fitting the learner puts three copies of the OLS estimator in the ``path``:
# one for each fold and one for the full dataset.
# These are named as ``[name].[col_id].[fold_id]``. To load these into the
# learner, we need to call ``collect``.

learner.collect(path)

############################################################################
# The main estimator, fitted on all data, gets stored into the
# ``learner_`` attribute, while the others are stored in the
# ``sublearners_``. These attributes are generators that create
# new sub-learners with fitted estimators when called upon.

############################################################################
# To generate predictions, we can either use the ``sublearners_``
# generator create cross-validated predictions, or ``learner_``
# generator to generate predictions for the whole input set.

############################################################################
# Similarly to above, we predict by specifying the job and the data to use.
# Now however, we must also specify the output array to populate.
# In particular, the learner will populate the columns given in the
# ``output_columns`` attribute, which is set with the ``setup`` call. If you
# don't want it to start populating from the first column, you can pass the
# ``n_left_concats`` argument to ``setup``. Here, we use the ``transform`` task,
# which uses the ``sublearners_`` generator to produce cross-validated
# predictions.

path = []
P = np.zeros((y.shape[0], 2))
learner.setup(X, y, 'transform', n_left_concats=1)
for sub_learner in learner.gen_transform(X, P):
    sub_learner.transform(path)
    print('Output:')
    print(P)
    print()

############################################################################
# In the above loop, a sub-segment of ``P`` is updated by each sublearner
# spawned by the learner. To instead produce predictions for the full
# dataset using the estimator fitted on all training data,
# task the learner to ``predict``.

############################################################################
# To streamline job generation across tasks and different classes, ML-Ensemble
# features a :class:`Job` class that manages job parameters.
# The job class prevents code repetition and allows us to treat the learner
# as a callable, enabling task-agnostic code:

job = Job(
    job='predict',
    stack=False,
    split=True,
    dir={},
    targets=y,
    predict_in=X,
    predict_out=np.zeros((y.shape[0], 1))
)

learner.setup(job.predict_in, job.targets, job.job)
for sub_learner in learner(job.args(), 'main'):
    sub_learner()
    print('Output:')
    print(job.predict_out)
    print()

############################################################################
# ML-Ensemble follows the Scikit-learn API, so if you wish to update any
# hyper-parameters of the estimator, use the ``get_params`` and ``set_params``
# API:

print("Params before:")
print(learner.get_params())

learner.set_params(estimator__offset=1, indexer__folds=3)

print("Params after:")
print(learner.get_params())

############################################################################
#
# .. note:: Updating the indexer on one learner updates the indexer on all
#  learners that where initiated with the same instance.

############################################################################
#
# Partitioning
# ------------
#
# We can create several other types of learners by
# varying the estimation strategy. An especially interesting strategy is to
# partition the training set and create several learners fitted on a given
# partition. This will create one prediction feature per partition.
# In the following example we fit the OLS model using two partitions and
# three fold CV on each partition. Note that by passing the output array
# as an argument during ``'fit'``, we perform a fit and transform operation.

from mlens.index import SubsetIndex

def mse(y, p): return np.mean((y - p) ** 2)

indexer = SubsetIndex(partitions=2, folds=2, X=X)
learner = Learner(estimator=OLS(),
                  indexer=indexer,
                  name='subsemble-ols',
                  scorer=mse,
                  verbose=True)

job.job = 'fit'
job.predict_out = np.zeros((y.shape[0], 2))

learner.setup(job.predict_in, job.targets, job.job)
for sub_learner in learner(job.args(), 'main'):
    sub_learner.fit()
    print('Output:')
    print(job.predict_out)
    print()

learner.collect()

############################################################################
# Each sub-learner records fit and predict times during fitting, and if
# a scorer is passed scores the predictions as well. The learner aggregates
# this data into a ``raw_data`` attribute in the form of a list.
# More conveniently, the ``data`` attribute returns a dict with a specialized
# representation that gives a tabular output directly:
# Standard data is fit time (``ft``), predict time (``pr``).
# If a scorer was passed to the learner, cross-validated test set prediction
# scores are computed. For brevity, ``-m`` denotes the mean and ``-s``
# denotes standard deviation.

print("Data:\n%s" % learner.data)

############################################################################
#
# Preprocessing
# -------------
#
# We can easily create a preprocessing pipeline before fitting the estimator.
# In general, several estimators will share the same preprocessing pipeline,
# so we don't want to pipeline the transformations in the estimator itself–
# this will result in duplicate transformers.

############################################################################
# As with estimators, transformers too define a computational sub-graph given
# a cross-validation strategy. Preprocessing pipelines are therefore wrapped
# by the :class:`Transformer` class, which is similar to the :class:`Learner`
# class. The input to the Transformer is a :class:`Pipeline` instance that holds the
# preprocessing pipeline.

############################################################################
#
# .. note::
#   When constructing a :class:`Pipeline` for use with the :class:`Transformer`,
#   the ``return_y`` argument must be ``True``.

############################################################################
# To link the transformer's sub-graph with the learner's sub-graph,
# we set the ``preprocess`` argument of the learner equal to the ``name``
# of the :class:`Transformer`. Note that any number of learners can share
# the same transformer and in fact should when the same preprocessing is desired.

from mlens.utils.dummy import Scale
from mlens.parallel import Transformer, Pipeline

pipeline = Pipeline([('trans', Scale())], return_y=True)

transformer = Transformer(estimator=pipeline,
                          indexer=indexer,
                          name='sc',
                          verbose=True)

############################################################################
# To build the learner we pass the ``name`` of the transformer as
# the ``preprocess`` argument:

learner = Learner(estimator=OLS(),
                  preprocess='sc',
                  indexer=indexer,
                  scorer=mse,
                  verbose=True)

###########################################################################
# We now repeat the above process to fit the learner, starting with fitting
# the transformer. By using the :class:`Job` class, we can write task-agnostic
# boiler-plate code. Note that the transformer is called as an
# ``'auxiliary'`` task, while the learner is called as the ``'main'`` task.

# Reset the prediction output array
job.predict_out = np.zeros((y.shape[0], 2))

transformer.setup(job.predict_in, job.targets, job.job)
learner.setup(job.predict_in, job.targets, job.job)

# Turn split off when you don't want the args() call to spawn a new sub-cache
job.split = False
for subtransformer in transformer(job.args(), 'auxiliary'):
    subtransformer()

for sublearner in learner(job.args(), 'main'):
    sublearner()

transformer.collect()
learner.collect()

############################################################################
# Note that the cache now contains the transformers as well:

print("Cache:")
for item in job.dir['task_%i' % job._n_dir]:
    print('{:20}{}'.format(*item))

############################################################################
# And estimation data is collected on a partition basis:

print("Data:\n%s" % learner.data)

############################################################################
#
# Parallel estimation
# -------------------
#
# Since the learner and transformer class do not perform estimations themselves,
# we are free to modify the estimation behavior. For instance, to parallelize
# estimation with several learners, we don't want a nested loop over each learner,
# but instead flatten the for loops for maximal concurrency.
# This is the topic of our next walk through. Here we show how to parallelize
# estimation with a single learner using multiple threads:

from multiprocessing.dummy import Pool

def run(est): est()

args = job.args()
job.predict_out = np.zeros((y.shape[0], 2))
job.job = 'predict'
Pool(4).map(run, list(learner(args, 'main')))

############################################################################
# For a slightly more high-level API for parallel computation on a single
# instance (of any accepted class), we can turn to the :func:`run` function.
# This function takes care of argument specification, array creation and all
# details we would otherwise need to attend to. For instance, to transform
# a dataset using the preprocessing pipeline fitted on the full training set,
# use :func:`run` to call ``predict``:

from mlens.parallel import run

print(
    run(transformer, 'predict', X)
)


############################################################################
# Next we handle several learners by grouping them in a layer in the
# :ref:`layer mechanics tutorial <layer_tutorial>`.

