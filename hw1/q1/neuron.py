#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Neuron:
    def __init__(self, threshold):
        self.threshold = threshold

    def is_active(self, value):
        if value >= self.threshold:
            return True
        return False


class ANDGate:
    def __init__(self, weight, threshold):
        self.weight = weight
        self.neuron = Neuron(threshold)
        assert threshold >= 2 * weight, "threshold must be two times the weight"

    def forward(self, x_1, x_2):
        val = (self.weight * x_1) + (self.weight * x_2)
        return self.neuron.is_active(val)


class ORGate:
    def __init__(self, weight, threshold):
        self.weight = weight
        self.neuron = Neuron(threshold)
        #  assert threshold == weight_pos, "threshold must be equal to weight"

    def forward(self, x_1, x_2):
        val = (self.weight * x_1) + (self.weight * x_2)
        return self.neuron.is_active(val)

class ANDNOTGate:
    def __init__(self, weight_pos, weight_neg, threshold):
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.neuron = Neuron(threshold)
        assert threshold == weight_pos, "threshold must be equal to positive weight"
        assert threshold >= weight_neg*-2, "threshold must be equal to positive weight"

    def forward(self, x_1, x_2):
        val = (self.weight_pos * x_1) + (self.weight_neg * x_2)
        return self.neuron.is_active(val)

class XORGate:
    def __init__(self, weight_pos, weight_neg, threshold):
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.and_not_gate = ANDNOTGate(weight_pos, weight_neg, threshold)
        self.or_gate = ORGate(weight_pos, threshold)

    def forward(self, x_1, x_2):
        z_1 = self.and_not_gate.forward(x_1, x_2)
        z_2 = self.and_not_gate.forward(x_2, x_1)
        return self.or_gate.forward(z_1, z_2)
