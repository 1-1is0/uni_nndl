#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
print(f"Python {sys.version}")
print("###############################################")

import tensorflow as tf
print(f"Tensor Flow Version: {tf.version}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
print()
