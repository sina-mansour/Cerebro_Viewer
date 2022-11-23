"""
This module contains the utility code to handle basic operations.

The goal is to put repetitive functions that are commonly used for throughout
the scripts in this module.

Notes
-----
Author: Sina Mansour L.
"""

import gc
import uuid


# Define a procedure to generate unique identifiers
def generate_unique_id():
    return uuid.uuid4()


# Define a function for garbage collection
def garbage_collect():
    gc.collect()
