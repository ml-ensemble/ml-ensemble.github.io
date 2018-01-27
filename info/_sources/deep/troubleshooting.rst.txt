.. Known issues

Troubleshooting
===============

Known potential issues. `Raise an issue`_ if your problem is not addressed here.

.. _third-party-issues:

Bad interaction with third-party packages
-----------------------------------------

ML-Ensemble itself is thread-safe, but third-party packages may not be. A known issue with Scikit-learn
(resolved as of 0.19.2) is that cloning is not thread-safe, estimators that clones internally 
(e.g. decision trees) can occasionally trigger an error. If ::

  IndexError: Pop from empty list

happens, try using ``multiprocessing`` instead.

With multiprocessing, be mindful of the ``start_method`` used. 
Due to how `Python forks the main process when running multiprocessing`_, 
workers can receive corrupted thread states prompting them to acquiring more threads than are available, 
with the resulting of a deadlock. Due to this limitation and the additional overhead of multiprocessing, 
If experiencing problems, try:

    #. ensure all estimators has ``n_jobs`` or ``nthread`` equal to ``1``,
    #. try changing the ``backend`` to either ``threading`` or ``multiprocessing``, 
    #. if using ``multiprocessing``, try varying the start method via :func:`~mlens.config.set_start_method`.
          
Changing the ``start_method`` from the default (``fork``) barrs the use of interactively defined
functions and classes (all functions and classes passed to an ``mlens`` object must be imported, not defined
the running script). For more information on multiprocessing issues see the `Scikit-learn FAQ`_.

Array copying during fitting
----------------------------

When the number of folds is greater than 2, it is not possible to slice the
full data in such a way as to return a view_ of that array (i.e. without
copying any data). Hence for fold numbers larger than 2, each worker 
copies a subset of the training data at estimation. If you experience memory-bound
issues, please consider using fewer folds during fitting. For further information on
avoiding copying data during estimation, see :ref:`memory`.

.. _GIL: https://wiki.python.org/moin/GlobalInterpreterLock
.. _view: http://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html
.. _Python forks the main process when running multiprocessing: https://wiki.python.org/moin/ParallelProcessing
.. _Scikit-learn FAQ: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
.. _issue tracker: https://github.com/flennerhag/mlens/issues
.. _Raise an issue: https://github.com/flennerhag/mlens/issues
