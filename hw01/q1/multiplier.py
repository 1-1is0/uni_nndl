#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import product

from neuron import ANDGate, ANDNOTGate, ORGate, XORGate

states = [0, 1]
ss = states
A = product(states, states)
B = product(states, states)


and_gate = ANDGate(1, 2)
and_not_gate = ANDNOTGate(weight_pos=2, weight_neg=-1, threshold=2)
or_gate = ORGate(2, 2)

print("AA", "BB", "CCCC", sep=" | ")
print("10", "10", "3210", sep=" | ")
print("--------------")
for a_1, a_0 in A:
    B = product(states, states)
    for b_1, b_0 in B:

        c_0 = int(and_gate.forward(a_0, b_0))
        z_0 = int(and_gate.forward(a_1, b_1))
        c_3 = int(and_gate.forward(c_0, z_0))
        c_2 = int(and_not_gate.forward(z_0, c_3))
        z_1 = int(and_gate.forward(a_1, b_0))
        z_2 = int(and_gate.forward(a_0, b_1))
        z_3 = int(or_gate.forward(z_1, z_2))
        c_1 = int(and_not_gate.forward(z_3, c_3))
        print(f"{a_1}{a_0}", f"{b_1}{b_0}", f"{c_3}{c_2}{c_1}{c_0}", sep=" | ")
