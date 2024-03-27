#' Elasticnet Regression
#'
#' @description Construct a elasticnet regression model with glmnet.
#'
#' @param alpha sequence of alpha values to search over
#'
#' @return A logistic elasticnet regression model for continuous outcomes,
#' with class 'learner'.
regr_elasticnet <- structure(
  class = c("learner_constructor", "function"),
  function(alpha = seq(0, 1, length.out = 5)) {

    make_learner(name        = "regr_elasticnet",
                 tune_fun    = purrr::partial(regr_elasticnet_tune,
                                              alpha = alpha),
                 predict_fun = regr_elasticnet_predict)
  }
)


# Methods -----------------------------------------------------------------
regr_elasticnet_tune <- function(features, tgt, wt = rep(1, nrow(features)),
                                 tune_folds, alpha) {

  tune_folds <- match(tune_folds, sort(unique(tune_folds)))

  cv_fits <- purrr::map(alpha,
                        ~glmnet::cv.glmnet(x = features,
                                           y = tgt,
                                           weights = wt,
                                           foldid  = tune_folds,
                                           alpha   = .x))

  tidy_fit <- function(fit, alpha) {

    data.frame(alpha    = alpha,
               lambda   = fit$lambda,
               loss      = fit$cvm)
  }

  hyperparams <- purrr::map2(cv_fits, alpha, tidy_fit)
  hyperparams = purrr::list_rbind(hyperparams)

  best_hyperparam_row <- which.min(hyperparams$loss)
  best_mod_idx        <- match(hyperparams$alpha[best_hyperparam_row], alpha)

  purrr::partial(regr_elasticnet_train,
                 lambda  = cv_fits[[best_mod_idx]]$lambda,
                 s       = hyperparams$lambda[best_hyperparam_row],
                 alpha   = hyperparams$alpha[best_hyperparam_row])
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

