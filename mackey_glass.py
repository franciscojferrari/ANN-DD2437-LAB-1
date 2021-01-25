#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:07:31 2021

@author: aleix

This program solves the mackey glass differential equation with the Euler method
It writes the solution into a file and it also plots the signal.
"""

import numpy as np
import matplotlib.pyplot as plt


class MackeyGlassSolver(object):
    def __init__(self, beta=0.2, gamma=0.1, n_coef=10, tau=25, x_init=1.5, 
                                                                     x_past=0):
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.x_init = x_init
        self.x_past = x_past
        self.n_coef = n_coef
    
    def solve_mackey_glass(self, t_init, t_end, dt):
        n_steps = int(np.round((t_end-t_init) / dt))
        time = np.linspace(t_init, t_end, n_steps+1)
        x_evol = np.zeros(n_steps+1)
        x_evol[0] = self.x_init # setting the initial condition
        for present_idx in range(n_steps):
            self.euler_step(x_evol, present_idx, dt)
        return time, x_evol
            
    def euler_step(self, x_evol, present_idx, dt):
        x_t_tau = self.get_delayed_term(x_evol, present_idx, dt)
        dxdt = self.beta*x_t_tau / (1+x_t_tau**self.n_coef) - self.gamma * \
                                                            x_evol[present_idx]
        x_evol[present_idx+1] = x_evol[present_idx] + dxdt*dt
        return
    
    # gets the value of x at time instant t-tau
    def get_delayed_term(self, x_evol, present_idx, dt):
        delayed_idx = present_idx - int(np.round(self.tau / dt))
        if delayed_idx < 0: 
            x_t_tau = self.x_past
        else:
            x_t_tau = x_evol[delayed_idx]
        return x_t_tau


def save_signal(time, signal, fname="mackey_glass.txt"):
    f = open(fname, "w")
    f.write("# time  mackey_glass\n")
    for i in range(time.size):
        f.write("%.16e    %.16e\n" % (time[i], signal[i]))
    f.close()
    return


t_init = 0
t_end = 1500
dt = 1

solver = MackeyGlassSolver()
time, x_evol = solver.solve_mackey_glass(t_init, t_end, dt)
save_signal(time, x_evol)

plt.figure()
plt.plot(time, x_evol) 










