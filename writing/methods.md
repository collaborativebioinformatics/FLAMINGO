# Methods

## Exact federated Mendelian randomization from sufficient statistics (FedMR)

Let site $k$ hold $n_k$ individuals with genotypes $G_k$ ($n_k \times m$
allele dosages), exposure $X_k$, outcome $Y_k$ and covariates $C_k$. Write
the structural regressors $W_k = [X_k, C_k]$ and the instruments
$Z_k = [G_k, C_k]$, each column centred within site so that a site
intercept is absorbed without being transmitted. Pooled two-stage least
squares over all sites is

$$
\hat\theta = \left(B^\top A^{-1} B\right)^{-1} B^\top A^{-1} c,
\qquad
A = \sum_k Z_k^\top Z_k,\;
B = \sum_k Z_k^\top W_k,\;
c = \sum_k Z_k^\top Y_k .
$$

Because every matrix is a sum of per-site cross-products, each site
computes and releases only $A_k$, $B_k$, $c_k$, together with
$D_k = W_k^\top W_k$, $e_k = W_k^\top Y_k$, $f_k = Y_k^\top Y_k$ and $n_k$;
the coordinator sums and solves. The residual sum of squares is
$f - 2\hat\theta^\top e + \hat\theta^\top D \hat\theta$, so the classical
covariance $\hat\sigma^2 (B^\top A^{-1} B)^{-1}$ with
$N - r - K$ residual degrees of freedom ($K$ absorbed site intercepts)
needs no further communication. A second round, in which the coordinator
broadcasts $\hat\theta$ and each site returns
$H_k = Z_k^\top \mathrm{diag}(\hat u_k^2) Z_k$, gives the
heteroskedasticity-robust covariance
$M^{-1} B^\top A^{-1} H A^{-1} B M^{-1}$ with $M = B^\top A^{-1} B$.
First-stage partial $F$ and partial $R^2$ of the excluded instruments are
functions of $A$, $B$ and $D$ alone.

Two protocols cover the two ways sites can hold instruments. With
harmonized SNPs (the same variants and effect allele everywhere),
$Z_k = [G_k, C_k]$ and every site contributes to every cell of $A$. When
each site's SNPs are its own variants, as in our simulation, there is no
shared first stage to learn: each site fits $X \sim [1, G, C]$ locally,
forms $\hat X_k$, and the second stage uses $Z_k = [\hat X_k, C_k]$, a
generated instrument whose cross-products all sites contribute to one
small matrix. This equals pooled 2SLS with per-site first stages and site
intercepts. A quadratic dose-response uses $W = [X, X^2, C]$ instrumented
by $[\hat X, \hat X^2, C]$; with a local first stage this stays at one
round, with shared SNPs it needs one round for the global first stage and
one for the second stage. A cross-fitted variant uses the out-of-fold
$\hat X$ as the instrument, solving
$\sum_i \hat X_i^{(-\mathrm{fold}(i))}(Y_i - \theta X_i) = 0$.

FedMR is distributed statistical estimation: the coordinator sees each
site's released matrices, which for the local-first-stage protocol are
seven scalars per site and for the shared protocol the site's within-site
LD matrix and GWAS-level sums. No privacy guarantee is claimed.

We implemented the estimator as a numpy package (`flamingo_fedmr`) and ran
it both in-process and as a two-round NVFlare workflow in which one client
per site computes its statistics and a server workflow sums and solves. On
every simulated continuous data set the NVFlare result, the in-process
result and the concatenated individual-level 2SLS agree to $10^{-16}$ in
the estimate and both standard errors.
