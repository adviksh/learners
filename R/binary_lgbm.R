#' Gradient Boosted Tree Classifier
#'
#' @description Construct a gradient boosted tree classifier using lightgbm
#' All parameters can be passed as single values, or as vectors. Performs
#' grid search over all combinations of provided values, using early stopping
#' to select the best number of boosting iterations.
#'
#' @return A gradient boosted tree model for binary outcomes,
#' with class 'learner'.
#'
#' @param learning_rate learning rate.
#' @param max_depth maximum depth of a tree.
#' @param bagging_fraction bagging_fraction ratio of observations when constructing each
#' tree. Values less than 1 will speed up computation, and may reduce
#' overfitting.
#' @param feature_fraction bagging_fraction ratio of columns when constructing
#' each tree. Values less than 1 will speed up computation, and may reduce
#' overfitting.
#' @param num_threads number of threads to use, to speed up training and prediction
#'
binary_lgbm <- structure(
  class = c("learner_constructor", "function"),
  function(learning_rate = 0.05,
           max_depth = c(0:6),
           bagging_fraction = 1,
           feature_fraction = 1,
           num_threads      = 1) {
    make_learner(name       = "binary_lgbm",
                 tune_fun    = purrr::partial(binary_lgbm_tune,
                                              learning_rate = learning_rate,
                                              max_depth = max_depth,
                                              bagging_fraction = bagging_fraction,
                                              feature_fraction = feature_fraction,
                                              num_threads = num_threads),
                 predict_fun = binary_lgbm_predict)
  }
)

# Methods -----------------------------------------------------------------
binary_lgbm_tune <- function(features, tgt, wt = rep(1, nrow(features)),
                             tune_folds, learning_rate, max_depth, bagging_fraction,
                             feature_fraction, num_threads) {

  binary_lgbm_design <- expand.grid(max_depth = max_depth,
                                    learning_rate = learning_rate,
                                    bagging_fraction = bagging_fraction,
                                    feature_fraction = feature_fraction)

  binary_lgbm_tuning <- purrr::pmap(binary_lgbm_design,
                                    function(...) {
                                      params = list(...)
                                      params$num_threads = num_threads
                                      params$objective   = 'binary'
                                      params$metric      = 'auc'
                                      params$verbosity   = -1

                                      lightgbm::lgb.cv(params = params,
                                                       data   = features,
                                                       label  = tgt,
                                                       weight = wt,
                                                       folds  = split(seq_along(tune_folds), tune_folds),
                                                       nrounds = 10000L,
                                                       early_stopping_rounds = 20L,
                                                       verbose = -1)
                                    })

  binary_lgbm_tuning_df <- purrr::map(binary_lgbm_tuning, tidy_xgb_cv)
  binary_lgbm_tuning_df = purrr::list_rbind(binary_lgbm_tuning_df)

  best_iter   <- which.min(binary_lgbm_tuning_df$rmse)

  purrr::partial(binary_lgbm_train,
                 params  = binary_lgbm_tuning_df$params[[best_iter]],
                 nrounds = binary_lgbm_tuning_df$nrounds[[best_iter]])
}

binary_lgbm_train <- function(features, tgt, wt, params, nrounds) {

  if (is.factor(tgt)) tgt <- as.integer(tgt) - 1L

  list(fit = lightgbm::lightgbm(data = features, label = tgt,
                                weight = wt,
                                params = params,
                                nrounds = nrounds,
                                verbose = -1),
       params = c(params, nrounds = nrounds))
}

binary_lgbm_predict <- function(model, features) {
  as.numeric(stats::predict(model$fit, features))
}

