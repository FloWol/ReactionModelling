import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# -----------------------------
# 1. Population and Signal Setup
# -----------------------------
def population_A(T):
    return np.exp(-0.05 * np.abs(T))

def population_C(T):
    return 0.2 * np.exp(-0.1 * np.abs(T))  # third state, less populated

T = np.linspace(-20, 20, 100)
T_bio = np.linspace(250, 350, 100)

popA = population_A(T)
popC = population_C(T)
pop_rest = 1 - popA - popC
popB = np.clip(pop_rest, 0, 1)  # ensure populations are valid

# Normalize populations
total = popA + popB + popC
popA /= total
popB /= total
popC /= total

# Plot populations
plt.figure()
plt.plot(T_bio, popA, label="popA")
plt.plot(T_bio, popB, label="popB")
plt.plot(T_bio, popC, label="popC")
plt.title("Three-State Populations")
plt.xlabel("Temperature (K)")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 2. Signal and True ΔG
# -----------------------------
sigA = 0.0448
sigB = 0.0340
sigC = 0.0250

sigT = popA * sigA + popB * sigB + popC * sigC

plt.figure()
plt.plot(T_bio, sigT)
plt.title("Total Signal $\sigma_T$")
plt.xlabel("Temperature (K)")
plt.ylabel("Signal")
plt.grid(True)
plt.show()

# Gibbs free energy (from true K = pB/pA)
R = 8.314
G_true = -R * T_bio * np.log(popA / popB) #fix this

plt.figure()
plt.plot(T_bio, G_true)
plt.title("True ΔG from $p_A/p_B$")
plt.xlabel("Temperature (K)")
plt.ylabel("ΔG (J/mol)")
plt.grid(True)
plt.show()

# -----------------------------
# 3. K Calculation and Shifting
# -----------------------------
def get_K_and_shift(sigA0, sigB0=0):
    K_vals = (sigA0 - sigT) / (sigT - sigB0)
    minK = np.min(K_vals)
    shift = -minK if minK < 0 else 0
    return K_vals + shift, shift, K_vals

# -----------------------------
# 4. Reference σ_A from sigT at multiple T
# -----------------------------
ref_indices = np.arange(0, len(T_bio), 10)
ref_labels = [f'{int(T_bio[i])}K' for i in ref_indices]

# Generate enough distinct colors
cmap = plt.get_cmap('viridis')
colors = [cmap(i / len(ref_indices)) for i in range(len(ref_indices))]

# -----------------------------
# 5. Plot Shifted K Curves
# -----------------------------
plt.figure(figsize=(7, 4), dpi=300)
plt.plot(T_bio, popB / popA, label='True $p_B/p_A$', color='black', linestyle='--')

# Base case with fixed sigmaA and sigmaB
K_base, shift_base, K_noshift = get_K_and_shift(sigA, sigB)
plt.plot(T_bio, K_base, label=r'Base $\sigma_A,\sigma_B$', color='tab:blue', linestyle='--')

shifts = []
K_no_shift_array = []

for idx, label, color in zip(ref_indices, ref_labels, colors):
    ref_sigmaA = sigT[idx]
    K_vals, shift, K_vals_no_shift = get_K_and_shift(ref_sigmaA, sigB)
    shifts.append((T_bio[idx], shift))
    plt.plot(T_bio, K_vals, label=fr'$\sigma_A$ at {label}', color=color)

plt.title("Shifted Equilibrium Constant $K$", fontsize=12)
plt.xlabel("Temperature (K)")
plt.ylabel("Shifted $K$")
plt.legend(fontsize=7, frameon=False, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
#   Non Shifted K for comparison
# -----------------------------

for idx, label, color in zip(ref_indices, ref_labels, colors):
    ref_sigmaA = sigT[idx]
    K_vals, shift, K_vals_no_shift = get_K_and_shift(ref_sigmaA, sigB)
    plt.plot(T_bio, K_vals_no_shift, label=fr'$\sigma_A$ at {label}', color=color)

plt.title("Equilibrium Constant $K$", fontsize=12)
plt.xlabel("Temperature (K)")
plt.ylabel("$K$")
plt.legend(fontsize=7, frameon=False, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Plot Shift vs. Reference Temperature
# -----------------------------
ref_temps, shift_vals = zip(*shifts)

plt.figure(figsize=(6, 3.5), dpi=300)
plt.plot(ref_temps, shift_vals, marker='o', linestyle='-', color='gray')
plt.plot(T_bio, popA, label="popA")
plt.plot(T_bio, popB, label="popB")
plt.plot(T_bio, popC, label="popC")
plt.title("Shift Required to Keep $K$ Positive", fontsize=12)
plt.xlabel("Reference Temperature for $\sigma_A$", fontsize=11)
plt.ylabel("Shift Value $C$", fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.show()

A_vals = []

plt.figure(figsize=(6, 3.5), dpi=300)
plt.plot(T_bio, K_base, label=r'Base $\sigma_A,\sigma_B$', color='tab:blue', ls="--")

for idx, label, color in zip(ref_indices, ref_labels, colors):
    ref_sigmaA = sigT[idx]
    K_vals, shift, K_vals_no_shift = get_K_and_shift(ref_sigmaA, sigB)
    # Minimize to find best A
    def cost(A):
        return np.sum((K_base - A * K_vals)**2)
    result = minimize_scalar(cost)
    A_opt = result.x
    A_vals.append(A_opt)
    # Apply the correction
    K_corrected = A_opt * K_vals



    plt.plot(T_bio, K_corrected, label=fr'$\sigma_A$ at {label}', color=color)

plt.title("corrected Equilibrium Constant $K$", fontsize=12)
plt.xlabel("Temperature (K)")
plt.ylabel("correct $K$")
plt.legend(fontsize=7, frameon=False, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 3.5), dpi=300)
plt.title(r"Slope correction from fit $A$", fontsize=12)
plt.xlabel("Temperature (K)")
plt.ylabel("$A$")
plt.plot(ref_temps, A_vals, marker='o', linestyle='-', color='gray')
plt.legend(fontsize=7, frameon=False, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()




