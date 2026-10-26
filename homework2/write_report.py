"""Generate the written solution and printable PDF after running solution.py."""
from pathlib import Path
import csv, json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
P=Path(__file__).resolve().parent
m=json.loads((P/'results/metrics.json').read_text())
rows=list(csv.reader((P/'results/coefficients.csv').open()))
text=r'''# Machine Learning and Optimization — Problem Set 2
## Problem 1: Ridge regression and cross-validation
Source: the supplied MLO_homework2.pdf and hitters.zip. All 263 observations and 19 features are used; no missing or nonfinite values were found. Salaries and coefficients are expressed in thousands of US dollars. The analysis uses NumPy, not a library ridge estimator or cross-validation routine.

## 1. Feature scaling
The features have very different scales: career totals can be thousands, annual counts are smaller, and indicator variables are 0 or 1. Ridge penalizes coefficient magnitudes, which depend on the measurement units. Without scaling, a large-scale feature needs a smaller coefficient for the same predictive effect and is consequently penalized less. Standardizing makes shrinkage comparable across features.
For each column j, compute mu_j = mean(X_j) and s_j = sqrt(mean((X_j - mu_j)^2)), then Z_ij = (X_ij - mu_j)/s_j. We use population standard deviations (ddof=0). All scales are positive. Centering is convenient with a free intercept and does not change the model class. The intercept column is added after scaling and is not standardized.
For the full-data coefficient path and final fit, these statistics use all 263 observations. During cross-validation they are computed using only the training fold, then applied unchanged to validation observations. This avoids leakage. Binary indicators are standardized too, following the instruction to scale all feature variables.

## 2. Closed-form solution with an unpenalized intercept
Let A = [1 Z], beta = (b, theta), and D = diag(0, 1, ..., 1). We minimize
    J(beta) = ||y - A beta||_2^2 + lambda * beta^T D beta.
The normal equations are (A^T A + lambda D) beta = A^T y, so
    beta_hat = (A^T A + lambda D)^(-1) A^T y.
The same expression applies to the assignment's augmented matrix [1 X]; here X has first been standardized. D leaves the intercept unpenalized. The implementation solves this linear system rather than explicitly forming the inverse. For lambda > 0 it has a unique solution: a zero quadratic form would require both theta = 0 and b*1 = 0.
Because Z has zero column means, b_hat = mean(y), and theta_hat = (Z^T Z + lambda I)^(-1) Z^T(y - mean(y)).
Convention: the squared-error loss above is a sum, with no division by n. Under the alternative objective mean squared error + alpha * ||theta||^2, alpha = lambda/n for the same training data. Numerical lambda values must be interpreted with this convention.

## 3. Coefficient norm along the regularization path
The grid is lambda_j = 10^(-3 + 10*j/99), j = 0, ..., 99. The first plot shows ||theta_hat||_2 against lambda with logarithmic axes; the intercept is excluded. All 99 successive differences in the norm are nonpositive within numerical tolerance.
This monotonicity also follows from optimality. If 0 < lambda_1 < lambda_2 and theta_1, theta_2 are the respective minimizers, adding their two optimality inequalities gives (lambda_2-lambda_1)(||theta_2||^2-||theta_1||^2) <= 0. Thus the norm cannot increase as lambda increases. The objective penalizes the squared norm; the requested plot displays the norm itself.

![Coefficient norm](results/coefficient_norm.png)

## 4. Small- and large-lambda limits
At lambda = 1e-8, the Euclidean difference between the augmented ridge and least-squares coefficients is SMALL_DIFF, a relative difference of SMALL_REL. Least squares is computed independently using numpy.linalg.lstsq. At the required grid endpoint lambda = 1e-3, the absolute coefficient difference is GRID_DIFF; the still smaller lambda demonstrates the limiting behavior more clearly.
At lambda = 1e12, ||theta_hat||_2 = LARGE_NORM, while b_hat = MEAN. At the grid endpoint lambda = 1e7, the slope norm is GRID_NORM. Thus all slopes approach zero and the prediction approaches the constant mean salary, rather than zero. In the centered formula, the eigenvalues of (Z^T Z + lambda I)^(-1) go to zero as lambda grows, proving the result.

## 5. Manual five-fold cross-validation
Use numpy.random.default_rng(42) to permute row indices, then numpy.array_split to create disjoint folds of sizes 53, 53, 53, 52, 52. For each lambda and fold, fit standardization on the other four folds, solve the ridge system on those training observations, and predict the held-out observations. No held-out target or feature statistic is used in fitting.
The score is pooled out-of-fold MSE: CV(lambda) = (1/263) * sum_i (y_i - prediction_i_from_held_out_fold(lambda))^2. This weights folds by their sizes. Choose the grid point with the smallest score, then refit using all observations and full-data scaling.
The selected lambda is BEST. Its cross-validation MSE is MSE (thousand dollars)^2, and RMSE is RMSE thousand dollars. The plot shows the complete grid. Fold assignments and all per-fold scores are saved for reproducibility. A different random partition or scaling convention can select a different grid point. This is a tuning score, not an unbiased independent test-set estimate after model selection; the exercise provides no separate test set.

![Cross-validation curve](results/cross_validation.png)

## 6. Final coefficients and coaching interpretation
The table below gives the refitted coefficients for standardized features and for original feature units. For a standardized feature, the coefficient is the fitted salary difference associated with a one-standard-deviation increase while holding other predictors fixed. For a binary feature, its original-units coefficient describes the full 0-to-1 change. The standardized intercept is the fitted salary at mean feature values.
The original-units slopes are theta_j/s_j and the original-units intercept is b - sum_j mu_j*theta_j/s_j. Both parameterizations give identical predictions.

COEFFICIENT_TABLE

A practical handful of attributes to emphasize would be: hitting reliably (Hits), producing runs over a sustained career (CRuns), reaching base through walks (Walks), and contributing defensively (PutOuts). Hits and CRuns have the largest positive standardized coefficients; Walks and PutOuts also have positive coefficients. Career hits, career RBI, and career home runs supply related evidence for sustained offensive production.
These are predictive associations, not causal coaching effects. Annual and career measures are strongly related, so individual coefficients can change as correlated predictors share explanatory power. For example, the negative AtBat coefficient does not mean a player should avoid batting; conditional on the same number of hits and other statistics, additional at-bats can reflect weaker efficiency. Similarly, negative career-walk or years coefficients should not be interpreted as evidence that walking or experience reduces pay. Career totals are accumulated outcomes, not independently manipulable skills. League and division indicators are context rather than player attributes. Coaching advice should therefore focus on the broad skills above, without promising a salary increase from this model.

## Reproduction and validation
Run `python solution.py` from this directory (or use its absolute path). Required runtime dependencies: NumPy and Matplotlib. To regenerate this report, run `python write_report.py` with ReportLab installed. No network service or credentials are needed.
Validation passed: expected input dimensions and finite values; strictly positive feature scales; monotone coefficient norms; tiny-lambda agreement with least squares; large-lambda shrinkage and mean intercept; exhaustive disjoint held-out folds; equality of standardized and original-units predictions; and agreement of the final coefficients with an independent centered singular-value-decomposition calculation.
'''
repl={'SMALL_DIFF':f"{m['small_lambda_coefficient_difference']:.8g}",'SMALL_REL':f"{m['small_lambda_relative_difference']:.8g}",'GRID_DIFF':f"{m['grid_small_lambda_difference']:.6g}",'LARGE_NORM':f"{m['large_lambda_slope_norm']:.8g}",'GRID_NORM':f"{m['grid_large_lambda_slope_norm']:.8g}",'MEAN':f"{m['mean_salary']:.8f}",'BEST':f"{m['best_lambda']:.10g}",'MSE':f"{m['cv_mse']:.6f}",'RMSE':f"{m['cv_rmse']:.6f}"}
for k,v in repl.items(): text=text.replace(k,v)
table=[['Feature','Standardized','Original units']]+[[r[0],f'{float(r[1]):.6f}',f'{float(r[2]):.6f}'] for r in rows[1:]]
mdtable='| Feature | Standardized coefficient | Original-units coefficient |\n|---|---:|---:|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in table[1:])
(P/'solution.md').write_text(text.replace('COEFFICIENT_TABLE',mdtable))
styles=getSampleStyleSheet(); story=[]
for para in text.split('\n\n'):
    para=para.strip()
    if not para: continue
    if para=='COEFFICIENT_TABLE':
        t=Table(table,repeatRows=1,colWidths=[150,150,150]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dde7f2')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('ALIGN',(1,1),(-1,-1),'RIGHT'),('BOTTOMPADDING',(0,0),(-1,-1),5)])); story.extend([t,Spacer(1,12)])
    elif para.startswith('!['):
        path=para.split('](')[1].rstrip(')'); story.extend([Image(str(P/path),width=450,height=289),Spacer(1,12)])
    else:
        for line in para.split('\n') if para.startswith('#') else [para]:
            level=len(line)-len(line.lstrip('#'))
            style=styles['Title'] if level==1 else styles['Heading2'] if level else styles['BodyText']
            story.append(Paragraph(escape(line.lstrip('# ').replace('`','')),style)); story.append(Spacer(1,7))
SimpleDocTemplate(str(P/'homework2_solution.pdf'),title='MLO Problem Set 2 - Worked Solution',author='',rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=42).build(story)
print('Created solution.md and homework2_solution.pdf')
