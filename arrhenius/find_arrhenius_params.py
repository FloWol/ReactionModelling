# Intensified random search for Arrhenius parameters producing the requested shapes.
# We'll run a larger random search and pick the best candidate according to a heuristic score.
# Then we plot the populations for the best candidate and print its parameters and diagnostics.
import numpy as np
import matplotlib.pyplot as plt

R = 8.31446261815324  # J/(mol*K)


def arrhenius(A, E, T):
    return A * np.exp(-E / (R * T))


def steady_state_populations(T, params):
    k_AB = arrhenius(params['A_AB'], params['E_AB'], T)
    k_BA = arrhenius(params['A_BA'], params['E_BA'], T)
    # k_AC = arrhenius(params['A_AC'], params['E_AC'], T)
    k_CB = arrhenius(params['A_CB'], params['E_CB'], T)
    # k_CA = arrhenius(params['A_CA'], params['E_CA'], T)
    k_BC = arrhenius(params['A_BC'], params['E_BC'], T)

    pA = np.zeros_like(T, dtype=float)
    pB = np.zeros_like(T, dtype=float)
    pC = np.zeros_like(T, dtype=float)

    for i in range(T.size):
        ka = k_AB[i];
        kb = k_BA[i];
        kf = k_BC[i];
        kd = k_CB[i]
        M = np.array([
            [-ka, kb, 0.0],
            [ka, -(kb + kf), kd],
            [0.0, kf, -kd]
        ], dtype=float)

        # Nullraum / stationäre Richtung bestimmen:
        # np.linalg.svd gibt uns den Nullraum (Eigenvektor zu kleinstem singulären Wert)
        U, S, Vt = np.linalg.svd(M)
        null_vec = Vt.T[:, -1]  # letzter Spaltenvektor entspricht Nullraum

        # Nur positive / normalisierte Lösung behalten
        p = np.abs(null_vec)
        p /= p.sum()

        pA[i], pB[i], pC[i] = p

    return pA, pB, pC


# Temperature grid
T_min, T_max = 200.0, 2000.0
T_vals = np.linspace(T_min, T_max, 300)


# Scoring function: higher is better
def score_candidate(pA, pB, pC):
    # 1) pA should decrease: prefer negative mean slope and small positive wiggles
    slope_pA = np.mean(np.diff(pA))
    monotonic_pA = -slope_pA  # more positive is better (decreasing => negative slope)
    wiggles_pA = np.sum(np.maximum(np.diff(pA), 0.0))  # positive increases penalize #kinda MC like
    score_pA = monotonic_pA - 5.0 * wiggles_pA

    # 2) pB should increase
    slope_pB = np.mean(np.diff(pB))
    wiggles_pB = np.sum(np.maximum(-np.diff(pB), 0.0))
    score_pB = slope_pB - 5.0 * wiggles_pB

    # 3) pC small at ends
    ends_small = - (pC[0] + pC[-1])  # more negative is better (we'll maximize overall)
    # 4) pC has a middle peak: peak value and location near middle
    peak_idx = np.argmax(pC)
    peak_val = pC.max()
    center_dist = abs(peak_idx - 0.5 * len(pC)) / (0.5 * len(pC))  # 0=center, 1=edge
    peak_center_score = peak_val * (1.0 - center_dist)
    # 5) prefer some prominence (peak significantly above ends)
    prominence = peak_val - 0.5 * (pC[0] + pC[-1])
    score_pC = peak_center_score + 2.0 * prominence + ends_small * 5.0

    # Combine scores with weights
    total = 2.0 * score_pA + 2.0 * score_pB + 3.0 * score_pC
    return total, {'score_pA': score_pA, 'score_pB': score_pB, 'score_pC': score_pC,
                   'slope_pA': slope_pA, 'slope_pB': slope_pB, 'peak_val': peak_val,
                   'peak_idx': peak_idx}


# Random search
np.random.seed(12345)
best = None
best_score = -1e9
best_diag = None
best_params = None
trials = 10000

for t in range(trials):
    params = {}
    # sample prefactor A log-uniform 1e9..1e15
    for pair in ['AB', 'BA', 'AC', 'CB', 'CA', 'BC']:
        params[f'A_{pair}'] = 10 ** np.random.uniform(9, 15)
        params[f'E_{pair}'] = np.random.uniform(1.5e4, 1.1e5)  # J/mol
    pA_vals, pB_vals, pC_vals = steady_state_populations(T_vals, params)
    total, diag = score_candidate(pA_vals, pB_vals, pC_vals)
    # penalize invalid (NaN or negative populations)
    if not np.all(np.isfinite(pA_vals)) or np.any(pA_vals < -1e-6):
        continue
    if total > best_score:
        best_score = total
        best = (pA_vals.copy(), pB_vals.copy(), pC_vals.copy())
        best_diag = diag
        best_params = params.copy()

# If nothing found (very unlikely), fallback
if best_params is None:
    best_params = {
        'A_AB': 1e12, 'E_AB': 9.0e4,
        'A_BA': 5e13, 'E_BA': 3.0e4,
        'A_AC': 5e11, 'E_AC': 8.5e4,
        'A_CB': 2e12, 'E_CB': 4.2e4,
        'A_CA': 1e11, 'E_CA': 1.0e5,
        'A_BC': 3e11, 'E_BC': 5.5e4
    }
    best = steady_state_populations(T_vals, best_params)
    best_score = None
    best_diag = None

pA_vals, pB_vals, pC_vals = best

# Plot best result
plt.figure(figsize=(9, 5))
plt.plot(T_vals, pA_vals, label='pA')
plt.plot(T_vals, pB_vals, label='pB')
plt.plot(T_vals, pC_vals, label='pC')
plt.xlabel('Temperature T [K]')
plt.ylabel('Stationary population')
plt.title('Best candidate from intensified search ({} trials)'.format(trials))
plt.legend()
plt.grid(True)
plt.tight_layout()

param_text = "\n".join([f"{k}: {v:.3g}" for k, v in best_params.items()])
plt.gcf().text(0.02, -0.02, "Best Arrhenius parameters (A in s^-1, E in J/mol):\n" + param_text, fontsize=8,
               va='bottom', ha='left')

plt.show()

# Print diagnostics and chosen parameters
print(best_score)
print(best_diag)
print(best_params)

'''
5.900361508643742
{'score_pA': np.float64(0.0033384338866823226), 'score_pB': np.float64(0.0031801311760173683), 'score_pC': np.float64(1.9624414595061146), 'slope_pA': np.float64(-0.0033384338866823226), 'slope_pB': np.float64(0.0031801311760173683), 'peak_val': np.float64(0.8740413895528099), 'peak_idx': np.int64(88)}
{'A_AB': 262509431789732.1, 'E_AB': 108562.50519432925, 'A_BA': 1328863052.786573, 'E_BA': 28130.84276403682, 'A_AC': 3724021277627.398, 'E_AC': 103324.75618809173, 'A_CB': 196986162303527.62, 'E_CB': 96909.07612266042, 'A_CA': 140854886466546.4, 'E_CA': 109779.64039019904, 'A_BC': 273910166907.9405, 'E_BC': 36995.29157457054}

'''