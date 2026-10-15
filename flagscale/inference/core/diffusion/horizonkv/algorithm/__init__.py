# Modified for the HorizonKV inference artifact; see NOTICE.
"""HorizonKV execution internals owned by the FlagScale integration.

The modules are intentionally grouped by responsibility:

* ``config`` and ``rasa_policy`` validate static and adaptive policies;
* ``oae`` and ``transfer`` implement KV residency/offload;
* ``attention`` and ``sparse_backend`` run sparse attention;
* ``pipeline`` provides the AR loop over the shared ``wan.backbone``.

Importing this package does not build or load a CUDA extension.
"""
__version__ = "0.1.0"
