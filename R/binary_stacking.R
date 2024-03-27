#' Stacking for Classification
#'
#' @description Construct a stacked model
#'
#' @return A stacking model for binary outcomes.
#'
#' @param learners a list of learners
#'
binary_stacking <- structure(
  class = c("learner_constructor", "function"),
  function(learners = list(elasticnet = binary_elasticnet(),
                           xgb = binary_xgb())) {
    make_learner(name        = "binary_stacking",
                 tune_fun    = purrr::partial(binary_stacking_tune,
                                              learners = learners),
                 predict_fun = binary_stacking_predict)
  }
)

# Methods -----------------------------------------------------------------
binary_stacking_tune <- function(features, tgt, wt = NULL, tune_folds, learners) {

  if (is.null(wt)) wt <- rep(1, length(tgt))

  # Tune Components ------------------------------------------------------------
  tuned_learners <- purrr::map(learners, tune,
                               features = features,
                               tgt = tgt,
                               wt = wt,
                               tune_folds = tune_folds)

  # Tune Metalearner -----------------------------------------------------------
  # Out-of-sample tuned predictions
  p_oos <- purrr::map(learners, tune_predict_oos,
                      features = features,
                      tgt = tgt,
                      wt  = wt,
                      tune_folds = tune_folds)
  p_oos <- Reduce(cbind, p_oos)

  colnames(p_oos) <- names(learners)

  trained_metalearner <- train(binary_convex_combo(),
                               features = p_oos,
                               tgt = tgt,
                               wt = wt)

  binary_stacking_train <- function(features, tgt, wt, learners, metalearner) {

    if (is.null(wt)) wt <- rep(1, length(tgt))

    trained_learners <- purrr::map(learners, train,
                                   features = features,
                                   tgt = tgt,
                                   wt = wt)

    fit <- list(learners    = trained_learners,
                metalearner = trained_metalearner)

    return(fit)
  }

  purrr::partial(binary_stacking_train,
                 learners    = tuned_learners,
                 metalearner = trained_metalearner)
}

binary_stacking_predict <- function(object, features) {

  p_comp <- purrr::map(object$learners, stats::predict, features)
  p_comp <- Reduce(cbind, p_comp)

  stats::predict(object$metalearner, p_comp)
}
