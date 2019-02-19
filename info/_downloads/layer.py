# -*- coding: utf-8 -*-
"""

.. _layer_tutorial:

.. currentmodule:: mlens.parallel

Layer Mechanics
===============

ML-Ensemble is designed to provide an easy user interface. But it is also designed
to be extremely flexible, all the wile providing maximum concurrency at minimal
memory consumption. The lower-level API that builds the ensemble and manages the
computations is constructed in as modular a fashion as possible.

The low-level API introduces a computational graph-like environment that you can
directly exploit to gain further control over your ensemble. In fact, building
your ensemble through the low-level API is almost as straight forward as using
the high-level API. In this tutorial, we will walk through how to use the
:class:`Group` and :class:`Layer` classes to fit several learners.

Suppose we want to fit several learners. The :ref:` learner tutorial <learner_tutorial`
showed us how to fit a single learner, and so one approach would be to simple
iterate over our learners and fit them one at a time. This however is a very slow
approach since we don't exploit the fact that learners can be trained in parallel.
Moreover, any type of aggregation, like putting all predictions into an array, would
have to be done manually.

The Layer API
^^^^^^^^^^^^^

To parallelize the implementation, we can use the :class:`Layer` class. A layer is
a handle that will run any number of :class:`Group` instances attached to it in parallel. Each
group in turn is a wrapper around a ``indexer-transformers-estimators`` triplet.

Basics
------

So, to fit our two learners in parallel, we first need a :class:`Group` object to
handle them.
"""
from mlens.parallel import Layer, Group, make_group, run
from mlens.utils.dummy import OLS, Scale
from mlens.index import FoldIndex


indexer = FoldIndex(folds=2)
group = make_group(indexer, [OLS(1), OLS(2)], None)

############################################################################
# This ``group`` object is now a complete description of how to fit our two
# learners using the prescribed indexing method.
#
# To train the estimators, we need feed the group to a :class:`Layer` instance:

import numpy as np

np.random.seed(2)

X = np.arange(20).reshape(10, 2)
y = np.random.rand(10)



layer = Layer(stack=group)

print(
    run(layer, 'fit', X, y, return_preds=True)
)

############################################################################
# To use some preprocessing before fitting the estimators, we can use the
# ``transformers`` argument when creating our ``group``:

group = make_group(indexer, [OLS(1), OLS(2)], [Scale()])

layer = Layer(stack=group)

print(
    run(layer, 'fit', X, y, return_preds=True)
)

############################################################################
#
# Multitasking
# ------------
#
# If we want our estimators two have different preprocessing, we can easily
# achieve this either by specifying different cases when making the group,
# or by making two separate groups. In the first case:


group = make_group(
    indexer,
    {'case-1': [OLS(1)], 'case-2': [OLS(2)]},
    {'case-1': [Scale()], 'case-2': []}
)

layer = Layer(stack=group)

print(
    run(layer, 'fit', X, y, return_preds=True)
)

############################################################################
# In the latter case:

groups = [
    make_group(indexer, OLS(1), Scale()), make_group(indexer, OLS(2), None)
]

layer = Layer(stack=groups)

print(
    run(layer, 'fit', X, y, return_preds=True)
)

############################################################################
# Which method to prefer depends on the application, but generally, it is
# preferable to put all transformers and all estimators belonging to a
# given indexing strategy into one ``group`` instance as it is easier to
# separate groups based on indexer and using cases to distinguish between
# different preprocessing pipelines.

############################################################################
# Now, suppose we want to do something more exotic, like using different
# indexing strategies for different estimators. This can easily be achieved
# by creating groups for each indexing strategy we want:

groups = [
    make_group(FoldIndex(2), OLS(1), Scale()),
    make_group(FoldIndex(4), OLS(2), None)
]

layer = Layer(stack=groups)

print(
    run(layer, 'fit', X, y, return_preds=True)
)


############################################################################
# Some care needs to be taken here: if indexing strategies do not return the
# same number of rows, the output array will be zero-padded.

from mlens.index import BlendIndex

groups = [
    make_group(FoldIndex(2), OLS(1), None),
    make_group(BlendIndex(0.5), OLS(1), None)
]

layer = Layer(stack=groups)
print(
    run(layer, 'fit', X, y, return_preds=True)
)

############################################################################
# Note that even if ``mlens`` indexer output different shapes, they preserve
# row indexing to ensure predictions are consistently mapped to their respective
# input. If you build a custom indexer, make sure that it uses a strictly
# sequential (with respect to row indexing) partitioning strategy.

############################################################################
#
# Layer features
# --------------
#
# A layer does not have to be specified all in one go; you can instantiate
# a layer and ``push`` and ``pop`` to its ``stack``.

layer = Layer()
group = make_group(FoldIndex(4), OLS(), None)
layer.push(group)

############################################################################
#
# .. note::
#
# If you push or pop to the stack, you must call ``fit`` before you can
# use the layer for prediction.

run(layer, 'fit', X, y)

group = make_group(FoldIndex(2), OLS(1), None)
layer.push(group)

try:
    run(layer, 'predict', X, y)
except Exception as exc:
    print("Error: %s" % str(exc))

############################################################################
# The :class:`Layer` class can print the progress of a job, as well as inspect
# data collected during the job. Note that the
# printouts of the layer does not take group membership into account.

from mlens.metrics import rmse

layer = Layer()
group1 = make_group(
    indexer,
    {'case-1': [OLS(1)], 'case-2': [OLS(2)]},
    {'case-1': [Scale()], 'case-2': []},
    learner_kwargs={'scorer': rmse}
)

layer.push(group1)

run(layer, 'fit', X, y, return_preds=True)
print()
print("Collected data:")
print(layer.data)
