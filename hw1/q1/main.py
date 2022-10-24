#!/usr/bin/env python
# -*- coding: utf-8 -*-

from  q1.neuron import ANDGate, ORGate, ANDNOTGate, XORGate

def test_gate(gate):
    print("x1", "x2", "y", sep=" | ")
    for x_1 in x_1_states:
        for x_2 in x_2_states:
            print(x_1, x_2, gate.forward(x_1, x_2),  sep="  |  ", )
    print()

x_1_states = (0, 1)
x_2_states = (0, 1)

print("AND Gate")
gate= ANDGate(1, 2)
test_gate(gate)


print("OR Gate")
gate= ORGate(2, threshold=2)
test_gate(gate)

print("AND NOT Gate")
gate = ANDNOTGate(weight_pos=2, weight_neg=-1, threshold=2)
test_gate(gate)

print("XOR Gate")
gate = XORGate(weight_pos=2, weight_neg=-1, threshold=2)
test_gate(gate)

