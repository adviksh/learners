#' Logistic elasticnet Regression
#' @export
#'
#' @description Construct a elasticnet regression model with glmnet.
#'
#' @param alpha sequence of alpha values to search over
#' @param workers integer number of cores to use for parallelized tuning
#'
#' @return A logistic elasticnet regression model for binary outcomes,
#' with class 'learner'.
binary_elasticnet = structure(
  class = c("learner_constructor", "function"),
  function(alpha   = seq(0, 1, length.out = 5),
           workers = 1) {

    make_learner(name        = "binary_elasticnet",
                 tune_fun    = purrr::partial(binary_elasticnet_tune,
                                              alpha   = alpha,
                                              workers = workers),
                 predict_fun = binary_elasticnet_predict,
                 predict_tuned_fun = binary_elasticnet_predict_tuned)
  }
)


# Methods -----------------------------------------------------------------
binary_elasticnet_tune = function(features, tgt, wt = rep(1, nrow(features)),
                                  tune_folds, alpha, workers) {

  tune_folds = match(tune_folds, sort(unique(tune_folds)))

  if (workers > 1) {
    parallel = TRUE
    doFuture::registerDoFuture()
    future::plan(future::multicore, workers = workers)
  } else {
    parallel = FALSE
  }

  max_vars = min(min(table(tune_folds)) - 1,
                 ncol(features))

  cv_fits = purrr::map(alpha,
                       ~glmnet::cv.glmnet(x = features,
                                          y = tgt,
                                          weights = wt,
                                          foldid  = tune_folds,
                                          family  = "binomial",
                                          alpha   = .x,
                                          keep    = TRUE,
                                          parallel = parallel,
                                          pmax     = max_vars))

  tidy_fit = function(fit, alpha) {

    data.frame(alpha    = alpha,
               lambda   = fit$lambda,
               loss      = fit$cvm)
  }

  extract_best_predictions = function(fits) {
    # Get the lowest MSE from each regression
    glmnet_cvm = purrr::map_dbl(fits, function(f) { min(f$cvm) })

    # Figure out which index in the output corresponds to predictions from the best fit
    best_fit = fits[[ which.min(glmnet_cvm) ]]
    lambda_min_idx = match(best_fit$lambda.min,
                           best_fit$lambda)

    # Return
    z_hat = best_fit$fit.preval[, lambda_min_idx]
    as.numeric(plogis(z_hat))
  }

  hyperparams = purrr::map2(cv_fits, alpha, tidy_fit)
  hyperparams = purrr::list_rbind(hyperparams)

  best_hyperparam_row = which.min(hyperparams$loss)
  best_mod_idx        = match(hyperparams$alpha[best_hyperparam_row], alpha)

  tune_res = list(
    tuned_model = extract_best_predictions(cv_fits),
    train_fun   = purrr::partial(binary_elasticnet_train,
                                 lambda  = cv_fits[[best_mod_idx]]$lambda,
                                 s       = hyperparams$lambda[best_hyperparam_row],
                                 alpha   = hyperparams$alpha[best_hyperparam_row])
  )

  tune_res
}

binary_elasticnet_train = function(features, tgt, wt, lambda, s, alpha) {
  list(fit = glmnet::glmnet(x = features,
                            y = tgt,
                            weights = wt,
                            family = "binomial",
                            alpha  = alpha,
                            lambda = lambda),
       alpha = alpha,
       s = s)
}

binary_elasticnet_predict = function(model, features) {
  as.numeric(stats::predict(model$fit, features,
                            s = model$s,
                            type = "response"))
}

binary_elasticnet_predict_tuned = function(model, features, tune_folds) {
  return(model)
}
