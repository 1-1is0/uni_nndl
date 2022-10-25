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