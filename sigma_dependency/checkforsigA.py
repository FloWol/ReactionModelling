# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:11:50 2025

@author: C9Wol
"""
import numpy as np
import matplotlib.pyplot as plt


def population(T,c):
    return np.exp(-0.05*np.abs(T))

T = np.linspace(-20, 20,100)
T_bio = np.linspace(250, 350,100)

popA = population(T, T[-1])
popB = 1-popA

plt.plot(T_bio, popA)
plt.plot(T_bio, popB)
plt.show()


G = -8.134*T_bio*np.log(popA/popB)
plt.plot(T_bio, G)
plt.show()


sigA = 0.04475523
sigB = 0.00403367

sigT = popA*sigA+popB*sigB
plt.plot(T_bio,sigT)
plt.show()

def K(sigA0, sigB0=0):
    return (sigA0 - sigT)/(sigT - sigB0)

# Plot setup
plt.figure(figsize=(6, 4), dpi=300)

# Plotting the curves
plt.plot(T_bio, popB/popA, label='True', color='black', linestyle='--')
plt.plot(T_bio, K(sigA), label=r'$\sigma_B=0$', color='tab:blue')
plt.plot(T_bio, K(sigT[0]), label=r'$\sigma_A$ at 250K', color='tab:orange')
plt.plot(T_bio, K(sigT[20]), label=r'$\sigma_A$ at 270K', color='tab:green')
plt.plot(T_bio, K(sigT[50]), label=r'$\sigma_A$ at 300K', color='tab:red')
plt.plot(T_bio, K(sigT[70]), label=r'$\sigma_A$ at 320K', color='tab:purple')

# Labels and title
plt.title("Temperature Dependence of Equilibrium Constant", fontsize=12)
plt.xlabel("Temperature (K)", fontsize=11)
plt.ylabel("Equilibrium Constant $K$", fontsize=11)

# Ticks
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Legend
plt.legend(fontsize=9, frameon=False)

# Tight layout for saving
plt.tight_layout()

# Show plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 8.314  # J/mol·K

# Calculate delta G from K
def delta_G(K, T):
    return -R * T * np.log(K)

# Plotting delta G instead of K
plt.title("Gibbs Free Energy ΔG vs. Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("ΔG (J/mol)")

plt.plot(T_bio, delta_G(popB/popA, T_bio), label='True')
plt.plot(T_bio, delta_G(K(sigA, sigB), T_bio), label='should be true')
plt.plot(T_bio, delta_G(K(sigA), T_bio), label='sigB=0')
plt.plot(T_bio, delta_G(K(sigT[0]), T_bio), label='sigT0')
plt.plot(T_bio, delta_G(K(sigT[20]), T_bio), label='sigT20')
plt.plot(T_bio, delta_G(K(sigT[50]), T_bio), label='sigT50')
plt.plot(T_bio, delta_G(K(sigT[70]), T_bio), label='sigT70')

plt.legend()
plt.grid(True)
plt.show()
