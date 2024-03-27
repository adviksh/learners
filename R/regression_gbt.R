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
regr_gbt <- structure(
  class = c("learner_constructor", "function"),
  function(eta = 0.05,
           max_depth = 1:6,
           subsample = 1,
           colsample_bytree = 1) {
    make_learner(name        = "regr_gbt",
                 tune_fun    = purrr::partial(regr_gbt_tune,
                                              eta = eta,
                                              max_depth = max_depth,
                                              subsample = subsample,
                                              colsample_bytree = colsample_bytree),
                 predict_fun = regr_gbt_predict)
  }
)

# Methods -----------------------------------------------------------------
regr_gbt_tune <- function(features, tgt, wt = rep(1, nrow(features)),
                          tune_folds, eta, max_depth, subsample,
                          colsample_bytree) {

  regr_gbt_design <- expand.grid(max_depth = max_depth,
                                 eta = eta,
                                 subsample = subsample,
                                 colsample_bytree = colsample_bytree)

  regr_gbt_tuning <- purrr::pmap(regr_gbt_design,
                                 xgboost::xgb.cv,
                                 data = xgboost::xgb.DMatrix(data = features,
                                                             label = tgt,
                                                             weight = wt),
                                 folds = split(seq_along(tune_folds),
                                               tune_folds),
                                 params  = list(nthread = 1L,
                                                objective = "reg:squarederror",
                                                eval.metric = "rmse",
                                                verbosity = 0),
                                 nrounds = 10000L,
                                 early_stopping_rounds = 20L,
                                 verbose = FALSE)

  regr_gbt_tuning_df <- purrr::map(regr_gbt_tuning, tidy_xgb_cv)
  regr_gbt_tuning_df = purrr::list_rbind(regr_gbt_tuning_df)

  best_iter   <- which.min(regr_gbt_tuning_df$rmse)

  purrr::partial(regr_gbt_train,
                 params  = regr_gbt_tuning_df$params[[best_iter]],
                 nrounds = regr_gbt_tuning_df$nrounds[[best_iter]])
}

regr_gbt_train <- function(features, tgt, wt, params, nrounds) {

  if (is.factor(tgt)) tgt <- as.integer(tgt) - 1L

  list(fit = xgboost::xgboost(data = features, label = tgt, weight = wt,
                              params = params,
                              nrounds = nrounds,
                              verbose = FALSE),
       params = c(params, nrounds = nrounds))
}

regr_gbt_predict <- function(model, features) {
  as.numeric(stats::predict(model$fit, features))
}
