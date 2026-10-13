"""HorizonKV execution internals owned by the FlagScale integration.

The modules are intentionally grouped by responsibility:

* ``config`` and ``rasa_policy`` validate static and adaptive policies;
* ``oae``, ``gpu_cache`` and ``transfer`` implement KV residency/offload;
* ``attention`` and ``sparse_backend`` run sparse attention;
* ``pipeline`` and ``backbone`` provide the model-independent AR loop.

Importing this package does not build or load a CUDA extension.
"""
__version__ = "0.1.0"
