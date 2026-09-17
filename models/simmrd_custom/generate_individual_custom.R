# Patches simmrd::generate_individual() so confounding is driven by one or
# more user-supplied confounders instead of a single internally-generated
# latent variable, and adds a wrapper that simulates a pair of biobank sites
# sharing the same genetic architecture and causal effect, differing only in
# which confounders each site observes -- matching the setup in
# summary_of_idea.md (Biobank A sees G,X,Y,C; Biobank B sees G,X,Y only).
#
# Requires: library(simmrd) (https://github.com/noahlorinczcomi/simmrd),
# mvnfast. Uses simmrd's unexported std()/parthcorr()/makeBlocks() helpers
# via `simmrd:::`.

`%||%` <- function(a, b) if (is.null(a)) b else a

#' One confounder spec for generate_individual_custom()
#'
#' @param generator function(n) -> numeric vector of length n. This is where
#'   you bake in a specific phenotype: draw from its empirical distribution,
#'   bootstrap-resample an observed vector, apply a known non-Gaussian family,
#'   etc. Default `function(n) rnorm(n)` reproduces simmrd's noise-only U.
#'   The result is standardized internally, so only its *shape* (not its
#'   mean/scale) matters.
#' @param effect_on_X proportion of variance explained in each exposure by
#'   this confounder (scalar, recycled, or a vector of length
#'   `number_of_exposures`).
#' @param effect_on_Y proportion of Y's variance explained by this confounder.
#' @param sign_on_Y +1 or -1 (default -1, matching simmrd's convention).
#' @param genetic_loading optional numeric vector, length
#'   `number_of_causal_SNPs`. If supplied, the confounder is
#'   `generator(n) + G %*% genetic_loading`, i.e. partly genetically driven
#'   (simmrd's CHP idea, generalized to any confounder). Omit for a
#'   confounder that is independent of G, which is what you want if your PRS
#'   independence certificate (S ⟂ C) is supposed to hold by construction.
#' @param observed_at character vector of site names where this confounder is
#'   returned in the output. It still affects X and Y at every site
#'   regardless of this setting -- it only controls what's visible, mirroring
#'   a real biobank that never collected the phenotype.
confounder_spec <- function(generator = function(n) stats::rnorm(n),
                             effect_on_X = 0, effect_on_Y = 0,
                             sign_on_Y = -1, genetic_loading = NULL,
                             observed_at = character(0)) {
  list(generator = generator, effect_on_X = effect_on_X, effect_on_Y = effect_on_Y,
       sign_on_Y = sign_on_Y, genetic_loading = genetic_loading, observed_at = observed_at)
}

#' Generate individual-level MR data with custom confounders
#'
#' Drop-in replacement for simmrd::generate_individual() individual-level
#' generation. `params$Xs_variance_explained_by_U`, `Y_variance_explained_by_U`,
#' `number_of_CHP_causal_SNPs`, `U_variance_explained_by_CHP` are ignored --
#' confounding is controlled entirely by `confounders`. UHP (SNPs with a
#' direct genetic effect on Y, bypassing X) is preserved from simmrd.
#'
#' @param params `simmrd::set_params(type = "individual", ...)` list.
#' @param confounders named list of `confounder_spec()` entries.
#' @param site character scalar identifying the current site, checked against
#'   each confounder's `observed_at`. NULL observes every confounder.
#' @param snp_effects optional architecture list from a previous call's
#'   `$snp_effects`, reused so multiple sites share the same B/theta/gammaU
#'   (the same underlying biology) while drawing independent individuals,
#'   noise, and confounder values.
#' @param seed RNG seed.
#' @return list(G, X, Y, C, theta, snp_effects)
generate_individual_custom <- function(params, confounders = list(), site = NULL,
                                        snp_effects = NULL, seed = 1) {
  set.seed(seed)
  for (i in seq_along(params)) assign(names(params)[i], params[[i]])

  if (length(Y_variance_explained_by_Xs) == 1)
    Y_variance_explained_by_Xs <- rep(Y_variance_explained_by_Xs, number_of_exposures)

  exposure_lengths <- c(number_of_exposures, length(Y_variance_explained_by_Xs), length(signs_of_causal_effects))
  if (exposure_lengths[1] < exposure_lengths[2] || exposure_lengths[1] < exposure_lengths[3]) {
    stop("`number_of_exposures`, `Y_variance_explained_by_Xs`, and `signs_of_causal_effects` must agree in length.")
  }

  n0_xy <- floor(prop_gwas_overlap_Xs_and_Y * min(sample_size_Y, sample_size_Xs))
  nall  <- sample_size_Y + sample_size_Xs - n0_xy
  maf   <- 0.3

  G <- matrix(stats::rbinom(number_of_causal_SNPs * nall, 2, maf), nrow = nall, ncol = number_of_causal_SNPs)
  G <- apply(G, 2, simmrd:::std)

  # --- genetic architecture: draw once, reuse across sites via snp_effects ---
  if (is.null(snp_effects)) {
    CorrXX    <- simmrd:::parthcorr(phenotypic_correlation_Xs, n = number_of_exposures)
    GenCorrXX <- simmrd:::parthcorr(genetic_correlation_Xs, n = number_of_exposures)
    LD        <- simmrd:::makeBlocks(LD_causal_SNPs, number_of_causal_SNPs, number_of_LD_blocks)
    K <- kronecker(GenCorrXX, LD)
    B <- matrix(mvnfast::rmvn(1, rep(0, nrow(K)), K), nrow = number_of_causal_SNPs, ncol = number_of_exposures)
    adj <- Xs_variance_explained_by_g / colSums(B^2)
    for (j in seq_len(ncol(B))) B[, j] <- sqrt(adj[j]) * B[, j]

    vXY <- Y_variance_explained_by_Xs
    theta <- vXY * signs_of_causal_effects
    adj <- vXY / sum(theta^2)
    adj[is.nan(adj)] <- 0
    theta <- sqrt(adj) * theta

    gammaU <- rep(0, number_of_causal_SNPs)
    if (number_of_UHP_causal_SNPs > 0) {
      uhpix <- 1:number_of_UHP_causal_SNPs
      gammaU_ <- stats::runif(number_of_UHP_causal_SNPs, -1, 1)
      adj <- Y_variance_explained_by_UHP / sum(gammaU_^2)
      gammaU[uhpix] <- sqrt(adj) * gammaU_
    }

    snp_effects <- list(B = B, theta = theta, gammaU = gammaU, CorrXX = CorrXX,
                         Xs_variance_explained_by_g = Xs_variance_explained_by_g,
                         Y_variance_explained_by_Xs = vXY,
                         Y_variance_explained_by_UHP = Y_variance_explained_by_UHP)
  }
  B <- snp_effects$B; theta <- snp_effects$theta; gammaU <- snp_effects$gammaU
  CorrXX <- snp_effects$CorrXX
  vg_X <- snp_effects$Xs_variance_explained_by_g
  vXY  <- snp_effects$Y_variance_explained_by_Xs
  vUHPY <- snp_effects$Y_variance_explained_by_UHP

  # --- confounders ---
  k <- length(confounders)
  Cmat <- matrix(0, nrow = nall, ncol = max(k, 1))
  pix  <- matrix(0, nrow = max(k, 1), ncol = number_of_exposures)
  piy  <- rep(0, max(k, 1))
  for (m in seq_len(k)) {
    spec  <- confounders[[m]]
    c_raw <- spec$generator(nall)
    if (!is.null(spec$genetic_loading)) c_raw <- c_raw + G %*% spec$genetic_loading
    Cmat[, m] <- simmrd:::std(c_raw)

    eff_x <- spec$effect_on_X
    if (length(eff_x) == 1) eff_x <- rep(eff_x, number_of_exposures)
    pix[m, ] <- sqrt(eff_x)
    piy[m] <- spec$sign_on_Y * sqrt(spec$effect_on_Y)
  }

  # --- exposures ---
  resid_var_X <- 1 - vg_X - colSums(pix^2)
  if (any(resid_var_X < 0)) stop("genetic + confounder variance exceeds 1 for at least one exposure")
  sdeX <- diag(sqrt(resid_var_X), number_of_exposures)
  SigmaEX <- sdeX %*% CorrXX %*% sdeX
  eX <- mvnfast::rmvn(nall, rep(0, number_of_exposures), SigmaEX)
  X  <- G %*% B + Cmat %*% pix + eX

  # --- outcome ---
  resid_var_Y <- 1 - sum(vXY) - vUHPY - sum(piy^2)
  if (resid_var_Y < 0) stop("exposure + UHP + confounder variance exceeds 1 for Y")
  eY <- stats::rnorm(nall, 0, sqrt(resid_var_Y))
  Y  <- X %*% theta + G %*% gammaU + Cmat %*% piy + eY

  observed <- vapply(confounders, function(spec) is.null(site) || site %in% spec$observed_at, logical(1))
  Cout <- NULL
  if (k > 0 && any(observed)) {
    Cout <- as.data.frame(Cmat[, observed, drop = FALSE])
    names(Cout) <- names(confounders)[observed]
  }

  list(G = G, X = X, Y = Y, C = Cout, theta = theta, snp_effects = snp_effects)
}

#' Simulate paired biobank sites sharing one causal architecture
#'
#' Draws the genetic architecture (SNP-exposure effects, causal effect of X
#' on Y, UHP pleiotropy) once, then calls generate_individual_custom() once
#' per site with independent individuals/noise/confounder draws and each
#' site's own sample size, dropping confounders that site doesn't observe.
#'
#' @param params shared `set_params(type = "individual", ...)` list.
#' @param confounders named list of `confounder_spec()` entries (their
#'   `observed_at` fields determine what each site sees).
#' @param site_sample_sizes named list, one entry per site, each a list with
#'   `sample_size_Xs` and `sample_size_Y` overrides for that site, e.g.
#'   `list(A = list(sample_size_Xs = 5000, sample_size_Y = 5000),
#'         B = list(sample_size_Xs = 20000, sample_size_Y = 20000))`.
#' @param seed base RNG seed; site i uses `seed + i`.
#' @return named list, one element per site, each `generate_individual_custom()`'s output.
simulate_federated_sites <- function(params, confounders, site_sample_sizes, seed = 1) {
  site_names <- names(site_sample_sizes)
  shared <- NULL
  out <- list()
  for (i in seq_along(site_names)) {
    s <- site_names[i]
    p <- params
    p[names(site_sample_sizes[[s]])] <- site_sample_sizes[[s]]
    res <- generate_individual_custom(p, confounders = confounders, site = s,
                                       snp_effects = shared, seed = seed + i)
    shared <- res$snp_effects
    out[[s]] <- res
  }
  out
}
