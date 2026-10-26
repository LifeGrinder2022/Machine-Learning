"""Problem Set 2. Run: python solution.py (NumPy and Matplotlib required)."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
X = np.loadtxt(ROOT / 'data/hitters.x.csv', delimiter=',', skiprows=1)
y = np.loadtxt(ROOT / 'data/hitters.y.csv', delimiter=',', skiprows=1)
with open(ROOT / 'data/hitters.x.csv') as f:
    names = next(csv.reader(f))
assert X.shape == (263, 19) and y.shape == (263,)
assert np.isfinite(X).all() and np.isfinite(y).all()

def scale_fit(x):
    mu, sd = x.mean(axis=0), x.std(axis=0, ddof=0)
    assert (sd > 0).all()
    return mu, sd

def ridge(z, target, lam):
    """Unnormalized squared loss + lam * ||slopes||^2; intercept unpenalized."""
    a = np.column_stack([np.ones(len(z)), z])
    penalty = np.diag([0.] + [1.] * z.shape[1])
    return np.linalg.solve(a.T @ a + lam * penalty, a.T @ target)

mu, sd = scale_fit(X)
Z = (X - mu) / sd
A = np.column_stack([np.ones(len(Z)), Z])
lambdas = np.logspace(-3, 7, 100)
path = np.array([ridge(Z, y, lam) for lam in lambdas])
norms = np.linalg.norm(path[:, 1:], axis=1)
assert np.all(np.diff(norms) <= 1e-8)
ols = np.linalg.lstsq(A, y, rcond=None)[0]
small = ridge(Z, y, 1e-8)
large = ridge(Z, y, 1e12)
assert np.allclose(small, ols, rtol=1e-6, atol=1e-5)
assert np.linalg.norm(large[1:]) < 1e-5
assert abs(large[0] - y.mean()) < 1e-7

# Explicit shuffled five-fold CV: each observation is held out exactly once.
rng = np.random.default_rng(42)
folds = np.array_split(rng.permutation(len(y)), 5)
fold_id = np.empty(len(y), dtype=int)
oof = np.empty((len(y), len(lambdas)))
fold_mse = np.empty((5, len(lambdas)))
for k, valid in enumerate(folds):
    train = np.setdiff1d(np.arange(len(y)), valid)
    fold_id[valid] = k
    m, s = scale_fit(X[train])
    ztrain, zvalid = (X[train]-m)/s, (X[valid]-m)/s
    aval = np.column_stack([np.ones(len(valid)), zvalid])
    for j, lam in enumerate(lambdas):
        pred = aval @ ridge(ztrain, y[train], lam)
        oof[valid, j] = pred
        fold_mse[k, j] = np.mean((y[valid]-pred)**2)
cv = np.mean((y[:, None]-oof)**2, axis=0)
assert sum(map(len, folds)) == len(y)
best = int(np.argmin(cv))
beta = path[best]
raw = beta[1:] / sd
raw_intercept = beta[0] - mu @ raw
assert np.allclose(A @ beta, raw_intercept + X @ raw)
# Independent centered-SVD implementation checks the fitted coefficients.
u, singular, vt = np.linalg.svd(Z, full_matrices=False)
svd_beta = vt.T @ ((singular/(singular**2+lambdas[best])) * (u.T @ (y-y.mean())))
assert np.allclose(beta[1:], svd_beta, rtol=1e-9, atol=1e-8)

fig, ax = plt.subplots(figsize=(7,4.5))
ax.loglog(lambdas, norms)
ax.set(xlabel=r'$\lambda$', ylabel=r'$\|\theta\|_2$ (intercept excluded)', title='Ridge coefficient norm')
ax.grid(True, which='both', alpha=.2)
fig.tight_layout(); fig.savefig(OUT/'coefficient_norm.png', dpi=180); plt.close(fig)
fig, ax = plt.subplots(figsize=(7,4.5))
ax.semilogx(lambdas, cv)
ax.axvline(lambdas[best], color='tab:red', linestyle='--', label=f'Best lambda = {lambdas[best]:.4g}')
ax.set(xlabel=r'$\lambda$', ylabel='Out-of-fold mean squared error', title='Manual five-fold cross-validation (seed 42)')
ax.legend(); ax.grid(True, alpha=.2)
fig.tight_layout(); fig.savefig(OUT/'cross_validation.png', dpi=180); plt.close(fig)
with open(OUT/'coefficients.csv','w') as f:
    w=csv.writer(f); w.writerow(['feature','standardized_coefficient','original_units_coefficient'])
    w.writerow(['Intercept',beta[0],raw_intercept])
    w.writerows(zip(names,beta[1:],raw))
np.savetxt(OUT/'cv_curve.csv', np.column_stack([lambdas,cv,fold_mse.T]), delimiter=',', header='lambda,pooled_mse,fold0_mse,fold1_mse,fold2_mse,fold3_mse,fold4_mse', comments='')
np.savetxt(OUT/'fold_assignment.csv', np.column_stack([np.arange(len(y)),fold_id]), delimiter=',', fmt='%d',header='row_zero_based,fold_zero_based',comments='')
metrics = dict(n=len(y), p=X.shape[1], best_lambda=float(lambdas[best]), cv_mse=float(cv[best]), cv_rmse=float(np.sqrt(cv[best])), intercept=float(beta[0]), small_lambda=1e-8, small_lambda_coefficient_difference=float(np.linalg.norm(small-ols)), small_lambda_relative_difference=float(np.linalg.norm(small-ols)/np.linalg.norm(ols)), large_lambda=1e12, large_lambda_slope_norm=float(np.linalg.norm(large[1:])), mean_salary=float(y.mean()), grid_small_lambda_difference=float(np.linalg.norm(path[0]-ols)), grid_large_lambda_slope_norm=float(norms[-1]))
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics, indent=2))
print('\nStandardized coefficients:')
for name, b in zip(names,beta[1:]): print(f'{name:12} {b:12.6f}')
print('All numerical validation checks passed.')
