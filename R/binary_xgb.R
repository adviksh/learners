#' Gradient Boosted Tree Classifier
#' @export
#' @description Construct a gradient boosted tree classifier using xgboost.
#' All parameters can be passed as single values, or as vectors. Performs
#' grid search over all combinations of provided values, using early stopping
#' to select the best number of boosting iterations.
#'
#' @return A gradient boosted tree model for binary outcomes,
#' with class 'learner'.
#'
#' @param eta learning rate.
#' @param max_depth maximum depth of a tree.
#' @param subsample subsample ratio of observations when constructing each
#' tree. Values less than 1 will speed up computation, and may reduce
#' overfitting.
#' @param colsample_bytree subsample ratio of columns when constructing
#' each tree. Values less than 1 will speed up computation, and may reduce
#' overfitting.
#'
binary_xgb <- structure(
  class = c("learner_constructor", "function"),
  function(eta = 0.05,
           max_depth = 0:6,
           subsample = 1,
           colsample_bytree = 1) {
    make_learner(name       = "binary_xgb",
                 tune_fun    = purrr::partial(binary_xgb_tune,
                                              eta = eta,
                                              max_depth = max_depth,
                                              subsample = subsample,
                                              colsample_bytree = colsample_bytree),
                 predict_fun = binary_xgb_predict)
  }
)

# Methods -----------------------------------------------------------------
binary_xgb_tune <- function(features, tgt, wt = rep(1, nrow(features)),
                            tune_folds, eta, max_depth, subsample,
                            colsample_bytree) {

  binary_xgb_design <- expand.grid(max_depth = max_depth,
                                   eta = eta,
                                   subsample = subsample,
                                   colsample_bytree = colsample_bytree)

  binary_xgb_tuning <- purrr::pmap(binary_xgb_design,
                                   xgboost::xgb.cv,
                                   data = xgboost::xgb.DMatrix(data = features,
                                                               label = tgt,
                                                               weight = wt),
                                   folds = split(seq_along(tune_folds),
                                                 tune_folds),
                                   params  = list(nthread = 1L,
                                                  objective   = "binary:logistic",
                                                  eval.metric = "rmse",
                                                  tree_method = 'approx',
                                                  verbosity   = 0),
                                   verbose = FALSE,
                                   nrounds = 10000L,
                                   early_stopping_rounds = 20L)

  binary_xgb_tuning_df <- purrr::map(binary_xgb_tuning, tidy_xgb_cv)
  binary_xgb_tuning_df = purrr::list_rbind(binary_xgb_tuning_df)

  best_iter   <- which.min(binary_xgb_tuning_df$rmse)

  purrr::partial(binary_xgb_train,
                 params  = binary_xgb_tuning_df$params[[best_iter]],
                 nrounds = binary_xgb_tuning_df$nrounds[[best_iter]])
}

binary_xgb_train <- function(features, tgt, wt, params, nrounds) {

  if (is.factor(tgt)) tgt <- as.integer(tgt) - 1L

  list(fit = xgboost::xgboost(data = features, label = tgt, weight = wt,
                              params = params,
                              verbose = FALSE,
                              nrounds = nrounds),
       params = c(params, nrounds = nrounds))
}

binary_xgb_predict <- function(model, features) {
  as.numeric(stats::predict(model$fit, features))
}

