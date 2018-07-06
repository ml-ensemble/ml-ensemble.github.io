# -*- coding: utf-8 -*-
"""

.. _parallel_tutorial:


.. currentmodule: mlens.parallel

Parallel Mechanics
==================

ML-Ensemble is designed to provide an easy user interface. But it is also designed
to be extremely flexible, all the wile providing maximum concurrency at minimal
memory consumption. The lower-level API that builds the ensemble and manages the
computations is constructed in as modular a fashion as possible.

The low-level API introduces a computational graph-like environment that you can
directly exploit to gain further control over your ensemble. In fact, building
your ensemble through the low-level API is almost as straight forward as using the
high-level API. In this tutorial, we will walk through the core
:class:`ParallelProcessing` class.

The purpose of the :class:`ParallelProcessing` class is to provide a streamlined
interface for scheduling and allocating jobs in a nested sequence of tasks. The
typical case is a sequence of :class:`Layer` instances where the output of one layer
becomes the input to the next. While the layers must therefore be fitted sequentially,
each layer should be fitted in parallel. We might be interested in propagating some of the
features from one layer to the next, in which case we need to take care of the array allocation.

ParallelProcessing API
^^^^^^^^^^^^^^^^^^^^^^

Basic map
¨¨¨¨¨¨¨¨¨

In the simplest case, we have a ``caller` that has a set of ``task``s that needs to be
evaluated in parallel. For instance, the ``caller`` might be a :class:`Learner`, with
each task being a fit job for a given cross-validation fold. In this simple case,
we want to perform an embarrassingly parallel for-loop of each fold, which we can
achieve with the ``map`` method of the :class:`ParallelProcessing` class.
"""
from mlens.parallel import ParallelProcessing, Job, Learner
from mlens.index import FoldIndex
from mlens.utils.dummy import OLS

import numpy as np

np.random.seed(2)

X = np.arange(20).reshape(10, 2)
y = np.random.rand(10)

indexer = FoldIndex(folds=2)
learner = Learner(estimator=OLS(),
                  indexer=indexer,
                  name='ols')

manager = ParallelProcessing(n_jobs=-1)

out = manager.map(learner, 'fit', X, y, return_preds=True)

print(out)

############################################################################
#
# Stacking a set of parallel jobs
# -------------------------------
#
# Suppose instead that we have a sequence of learners, where we want to fit
# each on the errors of the previous learner. We can achieve this by using
# ``stack`` method and a preprocessing pipeline for computing the errors.
# First, we need to construct a preprocessing class to transform the input,
# which will be the preceding learner's predictions, into errors.

from mlens.parallel import Transformer, Pipeline
from mlens.utils.dummy import Scale
from sklearn.base import BaseEstimator, TransformerMixin


def error_scorer(p, y):
    return np.abs(p - y)


class Error(BaseEstimator, TransformerMixin):

    """Transformer that computes the errors of a base learners"""

    def __init__(self, scorer):
        self.scorer = scorer

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        return self.scorer(X, y), y


############################################################################
# Now, we construct a sequence of tasks to compute, where the output of one
# task will be the input to the next. Hence, we want a sequence of the form
# ``[learner, transformer, learner, ..., transformer, learner]``:

tasks = []
for i in range(3):
    if i != 0:
        pipeline = Pipeline([('err', Error(error_scorer))], return_y=True)
        transformer = Transformer(
            estimator=pipeline,
            indexer=indexer,
            name='sc-%i' % (i + 1)
        )
        tasks.append(transformer)

    learner = Learner(
        estimator=OLS(),
        preprocess='sc-%i' % (i+1) if i != 0 else None,
        indexer=indexer,
        name='ols-%i' % (i + 1)
    )
    tasks.append(learner)


############################################################################
# To fit the stack, we call the ``stack`` method on the ``manager``, and since
# each learner must have access to their transformer, we set ``split=False``
# (otherwise each task will have a separate sub-cache, sealing them off
# from each other).

out = manager.stack(
    tasks, 'fit', X, y, return_preds=True, split=False)

print(out)

############################################################################
# If we instead want to append these errors as features, we can simply
# alter our transformer to concatenate the errors to the original data.
# Alternatively, we can automate the process by instead using the
# :class:`mlens.ensemble.Sequential`  API.

############################################################################
#
# Manual initialization and processing
# ------------------------------------
#
# Under the hood, both ``map`` and ``stack`` first call ``initialize`` on the
# ``manager``, followed by a call to ``process`` with some default arguments.
# For maximum control, we can manually do the initialization and processing step.
# When we initialize, an instance of :class:`Job` is created that collect arguments
# relevant for of the job as well as handles for data to be used. For instance,
# we can specify that we want the predictions of all layers, as opposed to just the
# final layer:

out = manager.initialize(
    'fit', X, y, None, return_preds=['ols-1', 'ols-2'], stack=True, split=False)

############################################################################
# The ``initialize`` method primarily allocates memory of input data and
# puts it on the ``job`` instance. Not that if the input is a string pointing
# to data on disk, ``initialize`` will attempt to load the data into memory.
# If the backend of the manger is ``threading``, keeping the data on the parent
# process is sufficient for workers to reach it. With ``multiprocessing`` as
# the backend, data will be memory-mapped to avoid serialization.

############################################################################
# The ``initialize`` method returns an ``out`` dictionary that specified
# what type of output we want when running the manager on the assigned job.
# To run the manager, we call ``process`` with out ``out`` pointer:

out = manager.process(tasks, out)
print(out)

############################################################################
# The output now is a list of arrays, the last of which is the same predicitons
# as we got when using ``stack``.

############################################################################
#

# Memory management
# -----------------
#
# When running the manager, it will read and write to memory buffers. This is
# less of a concern when the ``threading`` backend is used, as data is kept
# in the parent process. But when data is loaded from file path, or when
# ``multiprocessing`` is used, we want to clean up after us. Thus, when we
# are through with the ``manager``, it is important to call the ``clear``
# method. This will however destroy any ephemeral data stored on the instance.

manager.clear()

############################################################################
#
# ..warning:: The ``clear`` method will remove any files in the specified path.
# If the path specified in the ``initialize`` call includes files other than
# those generated in the ``process`` call, these will ALSO be removed.
# ALWAYS use a clean temporary cache for processing jobs.

############################################################################
# To minimize the risk of forgetting this last step, the :class:`ParallelProcessing`
# class can be used as context manager, automatically cleaning up the cache
# when exiting the context:

learner = Learner(estimator=OLS(), indexer=indexer)

with ParallelProcessing() as mananger:
    manager.stack(learner, 'fit', X, y, split=False)
    out = manager.stack(learner, 'predict', X, split=False)
