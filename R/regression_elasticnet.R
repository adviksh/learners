#' Elasticnet Regression
#' @export
#'
#' @description Construct a elasticnet regression model with glmnet.
#'
#' @param alpha sequence of alpha values to search over
#'
#' @return A logistic elasticnet regression model for continuous outcomes,
#' with class 'learner'.
regr_elasticnet <- structure(
  class = c("learner_constructor", "function"),
  function(alpha = seq(0, 1, length.out = 5), workers = 1) {

    make_learner(name        = "regr_elasticnet",
                 tune_fun    = purrr::partial(regr_elasticnet_tune,
                                              alpha = alpha,
                                              workers = workers),
                 predict_fun = regr_elasticnet_predict,
                 predict_tuned_fun = regr_elasticnet_predict_tuned)
  }
)


# Methods -----------------------------------------------------------------
regr_elasticnet_tune <- function(features, tgt, wt = rep(1, nrow(features)),
                                 tune_folds, alpha, workers) {

  tune_folds <- match(tune_folds, sort(unique(tune_folds)))

  if (workers > 1) {
    parallel = TRUE
    doFuture::registerDoFuture()
    future::plan(future::multicore, workers = workers)
  } else {
    parallel = FALSE
  }


  cv_fits = purrr::map(alpha,
                       function(aa) {
                         glmnet::cv.glmnet(x        = features,
                                           y        = tgt,
                                           weights  = wt,
                                           foldid   = tune_folds,
                                           alpha    = aa,
                                           keep     = TRUE,
                                           parallel = parallel)
                       })

  tidy_fit = function(fit, alpha) {

    data.frame(alpha    = alpha,
               lambda   = fit$lambda,
               loss     = fit$cvm)
  }

  extract_best_predictions = function(fits) {
    # Get the lowest MSE from each regression
    glmnet_cvm = purrr::map_dbl(fits, function(f) { min(f$cvm) })

    # Figure out which index in the output corresponds to predictions from the best fit
    best_fit = fits[[ which.min(glmnet_cvm) ]]
    lambda_min_idx = match(best_fit$lambda.min,
                           best_fit$lambda)

    # Return
    best_fit$fit.preval[, lambda_min_idx]
  }

  hyperparams = purrr::map2(cv_fits, alpha, tidy_fit)
  hyperparams = purrr::list_rbind(hyperparams)

  best_hyperparam_row = which.min(hyperparams$loss)
  best_mod_idx        = match(hyperparams$alpha[best_hyperparam_row], alpha)

  tune_res = list(
    tuned_model = extract_best_predictions(cv_fits),
    train_fun   = purrr::partial(regr_elasticnet_train,
                                 lambda  = cv_fits[[best_mod_idx]]$lambda,
                                 s       = hyperparams$lambda[best_hyperparam_row],
                                 alpha   = hyperparams$alpha[best_hyperparam_row])
  )

  tune_res
}

regr_elasticnet_train <- function(features, tgt, wt, lambda, s, alpha) {
  list(fit = glmnet::glmnet(x = features,
                            y = tgt,
                            weights = wt,
                            alpha = alpha,
                            lambda = lambda),
       alpha = alpha,
       s = s)
}

regr_elasticnet_predict <- function(model, features) {
  as.numeric(stats::predict(model$fit, features,
                            s = model$s,
                            type = "response"))
}

regr_elasticnet_predict_tuned = function(model, features, tune_folds) {
  return(model)
}
