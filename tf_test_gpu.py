#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
print(f"Python {sys.version}")
print("###############################################")

import torch
print(f"Torch Version: {torch.version}")
print(f"Torch GPU: {torch.cuda.is_available()}")
print(f"Torch GPU Name: {torch.cuda.get_device_name()}")
print("###############################################")

import tensorflow as tf
print(f"Tensor Flow Version: {tf.version}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
print()
