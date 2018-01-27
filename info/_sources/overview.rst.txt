.. Overview of tutorials

 .. _mechanics-overview:

Overview
========

To build computational graphs, you need to specify an ``estimator``, a (set of) indexer(s), and any preprocessing pipelines wanted. 
The indexer(s) determine the cross validation strategy to use during ``fit`` and ``transform`` calls. The :class:`~mlens.index.FullIndex` can
be used if no cross-validation is wanted. 

.. currentmodule:: mlens.parallel

The basic node is the :class:`Learner`. The learner contains the sub-graph pertaining to a specific indexer-estimator-preprocessing configuration.
Similarly, the :class:`Transformer` contains the sub-graph of a specific indexer-preprocessing configuration. To learn more, see the 
`learner mechanics <learner_tutorial>`_ tutorial.

To build a graph of a set of :class:`Learner`` and :class:`Transformer` instances, several handles come in handy for ensuring efficient computation.
The :class:`Group` class is a handle for a set of learners and transformers sharing a specific indexer, while the :class:`Layer` class is a handle for
a set of groups. To learn more about how to efficiently parallelize independent estimations, see the `layer mechanics <layer_tutorial>`_ tutorial.

Frequently, we want the output of some set of learners to be feed as input to some other set. The :class:`ParallelProcessing` engine is a purpose-built
parallel job-execution engine design to handle these types of scenarios. Together with the :class:`~mlens.ensemble.base.Sequential` handle, designing 
deep computational graphs is straightforward. See the `advanced graph mechanics <parallel_tutorial>`_ tutorial for more information.
