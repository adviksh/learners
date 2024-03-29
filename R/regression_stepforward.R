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
    future::plan(future::multicore, workers = workers)
  } else {
    parallel = FALSE
  }

  cv_fits = regr_regsubsets_cv(x = features,
                               y = tgt,
                               weights  = wt,
                               parallel = parallel)

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
                           nvmax   = n_cols)

  smr = leaps:::summary.regsubsets(fits)

  fit_coef     = stats::coef(fits, n_cols)
  fit_coef_idx = smr$which[n_cols,]

  model_coef = rep(0, ncol(features) + 1)
  names(model_coef) = c('(Intercept)', colnames(features))

  model_coef[fit_coef_idx] = fit_coef

  return(model_coef)
}

regr_stepforward_predict = function(model, features) {
  as.numeric(model[1] + (features %*% model[-1]))
}

regr_stepforward_predict_tuned = function(model, features, tune_folds) {
  return(model)
}

regr_regsubsets_cv = function(x, y, weights = rep(1, length(y)), parallel = FALSE) {

  # fit forward regression
  fits_leaps = leaps::regsubsets(x = x,
                                 y = y,
                                 weights = weights,
                                 method = 'forward',
                                 nbest = 1,
                                 nvmax = min(ncol(x), nrow(x) - 2))
  smr = summary(fits_leaps)

  # extract coefficient path
  coef_cols = apply(smr$which[,-1], 1, \(x) as.numeric(which(x)))

  # get leave-one-out error
  if (parallel) {
    if (rlang::is_installed("furrr") == FALSE) {
      stop("Running regr_regsubsets_cv in parallel requires the package 'furrr'")
    }
    mapper = furrr::future_map
  } else {
    mapper = purrr::map
  }

  path_loocv = mapper(coef_cols,
                      function(which_cols, x, y, w) {
                        xs  = x[, which_cols, drop = FALSE]
                        fit = lm(y ~ xs, weights = w, model = FALSE)
                        hat = hatvalues(fit)

                        rsd_loo = fit$residuals / (1 - hat)

                        list(n_cols = length(which_cols),
                             mse    = mean((rsd_loo)^2),
                             y_hat  = y - rsd_loo)
                      },
                      x = x, y = y, w = weights)

  return(path_loocv)
}
