"""MTMD/llama.cpp helpers used by the selected Task 2 baseline."""

from .mtmd_client import MtmdRerankerClient, MtmdRequestTimeout, MtmdWorkerError

# The frozen benchmark uses MTMD only.  Keep this exception name so the Task 2
# runner can report a clear error if an unsupported QAIRT-reranker branch is selected.
class QairtWorkerError(RuntimeError):
    pass

__all__ = ["MtmdRerankerClient", "MtmdRequestTimeout", "MtmdWorkerError", "QairtWorkerError"]
