"""
Server entry point — patches multiprocess.ResourceTracker before
uvicorn imports any application code.

Usage:  python start_server.py
"""
import os, sys

# ── Crash prevention — MUST be before ANY other imports ──
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Monkey-patch multiprocess ResourceTracker BEFORE uvicorn/torch/SBERT
# are imported.  Its __del__ calls _stop() which kills the parent
# process with a silent exit(1) on Windows.
try:
    import multiprocess.resource_tracker as _mrt
    _mrt.ResourceTracker.__del__ = lambda self: None
except Exception:
    pass

import torch
torch.set_num_threads(1)

# Ensure app/server is on sys.path so uvicorn can find `main`
server_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "server")
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:socket_app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
