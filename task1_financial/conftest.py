"""
Pytest configuration and root sys.path initialization.
Includes Python 3.10+ backward-compatibility shims for legacy dependencies.
"""

import collections
import collections.abc
import os
import sys
from pathlib import Path

# Python 3.10+ compatibility shim for legacy packages referencing collections.Mapping
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping

if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping

# Add task1_financial root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
