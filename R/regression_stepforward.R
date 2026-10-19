#' Forward Stepwise OLS Regression
#' @export
#'
#' @return A linear regression model for continuous outcomes,
#' with class 'learner'.
regr_stepforward <- structure(
  class = c("learner_constructor", "function"),
  function(workers = 1) {
    make_learner(
      name = "regr_stepforward",
      tune_fun          = purrr::partial(regr_stepforward_tune,
                                         workers = workers),
      train_fun         = regr_stepforward_train,
      predict_fun       = regr_stepforward_predict,
      predict_tuned_fun = regr_stepforward_predict_tuned
    )
  }
)

regr_stepforward_tune = function(features, tgt, wt = rep(1, nrow(features)), tune_folds, workers) {

  if (workers > 1) {
    parallel = TRUE
    oplan = future::plan(future::multicore, workers = workers)
    on.exit(future::plan(oplan), add = TRUE)
  } else {
    parallel = FALSE
  }

  cv_fits = regr_regsubsets_cv(x = features,
                               y = tgt,
                               weights    = wt,
                               tune_folds = tune_folds,
                               parallel   = parallel)

  best_idx = which.min(purrr::map_dbl(cv_fits, "mse"))

  tune_res = list(
    tuned_model = cv_fits[[best_idx]]$y_hat,
    train_fun   = purrr::partial(regr_stepforward_train,
                                 n_cols = cv_fits[[best_idx]]$n_cols)
  )

  tune_res

}

regr_stepforward_train = function(features, tgt, wt, n_cols) {

  fits = leaps::regsubsets(x = features,
                           y = tgt,
                           weights = wt,
                           method  = 'forward',
                           nbest   = 1,
                           nvmax   = min(n_cols, ncol(features), nrow(features) - 2))

  smr = summary(fits)

  # the path may be shorter than requested
  size = min(n_cols, nrow(smr$which))
  fit_cols = which(smr$which[size, -1])

  model_coef = rep(0, ncol(features) + 1)
  names(model_coef) = c('(Intercept)', colnames(features))

  fit_coef = stats::coef(fits, size)
  model_coef[c(1, fit_cols + 1)] = unname(fit_coef)

  return(model_coef)
}

regr_stepforward_predict = function(model, features) {
  as.numeric(model[1] + (features %*% model[-1]))
}

regr_stepforward_predict_tuned = function(model, features, tune_folds) {
  return(model)
}

# Forward selection on the rows in `train_idx`; returns out-of-sample
# predictions for the rows in `pred_idx` at every model size 1:nvmax
# (a matrix with one column per size).
regr_regsubsets_cv_fold = function(which_fold, x, y, w, tune_folds, nvmax) {

  train_idx = which(tune_folds != which_fold)
  pred_idx  = which(tune_folds == which_fold)

  fits = leaps::regsubsets(x = x[train_idx, , drop = FALSE],
                           y = y[train_idx],
                           weights = w[train_idx],
                           method  = 'forward',
                           nbest   = 1,
                           nvmax   = nvmax)
  smr = summary(fits)

  preds = matrix(NA_real_, nrow = length(pred_idx), ncol = nvmax)

  for (ss in seq_len(min(nvmax, nrow(smr$which)))) {
    cols = which(smr$which[ss, -1])
    xs   = cbind(1, x[train_idx, cols, drop = FALSE])
    beta = stats::lm.wfit(xs, y[train_idx], w[train_idx])$coefficients
    beta[is.na(beta)] = 0

    preds[, ss] = as.numeric(cbind(1, x[pred_idx, cols, drop = FALSE]) %*% beta)
  }

  # if the path ended early, carry the last model forward
  last = min(nvmax, nrow(smr$which))
  if (last < nvmax) {
    preds[, (last + 1):nvmax] = preds[, last]
  }

  list(pred_idx = pred_idx, preds = preds)
}

# Honest K-fold CV over `tune_folds`: variable selection is redone inside each
# training fold. Returns one element per model size with its CV mse,
# out-of-fold predictions (y_hat, in original row order), and size (n_cols).
regr_regsubsets_cv = function(x, y, weights = rep(1, length(y)),
                              tune_folds, parallel = FALSE) {

  fold_vals = sort(unique(tune_folds))
  fold_n    = as.numeric(table(tune_folds))

  # same size cap for every fold
  nvmax = max(1, min(ncol(x), nrow(x) - max(fold_n) - 2))

  if (parallel) {
    if (rlang::is_installed("furrr") == FALSE) {
      stop("Running regr_regsubsets_cv in parallel requires the package 'furrr'")
    }
    fold_res = furrr::future_map(fold_vals, regr_regsubsets_cv_fold,
                                 x = x, y = y, w = weights,
                                 tune_folds = tune_folds, nvmax = nvmax)
  } else {
    fold_res = purrr::map(fold_vals, regr_regsubsets_cv_fold,
                          x = x, y = y, w = weights,
                          tune_folds = tune_folds, nvmax = nvmax)
  }

  oof = matrix(NA_real_, nrow = length(y), ncol = nvmax)
  for (fr in fold_res) {
    oof[fr$pred_idx, ] = fr$preds
  }

  purrr::map(seq_len(nvmax), function(ss) {
    list(n_cols = ss,
         mse    = stats::weighted.mean((y - oof[, ss])^2, weights),
         y_hat  = oof[, ss])
  })
}
