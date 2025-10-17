#' Gradient Boosted Tree Classifier
#' @export
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
#' @param feature_fraction bagging_fraction ratio of columns when constructing
#' each tree. Values less than 1 will speed up computation, and may reduce
#' overfitting.
#' @param num_threads number of threads to use, to speed up training and prediction
#'
multiclass_lgbm = structure(
  class = c("learner_constructor", "function"),
  function(learning_rate = 0.05,
           num_leaves = 31,
           max_depth  = -1,
           feature_fraction = 1,
           force_col_wise = FALSE,
           metric  = 'auc_mu',
           nrounds = 10000L,
           workers = 1,
           n_class) {
    make_learner(name       = "multiclass_lgbm",
                 tune_fun    = purrr::partial(multiclass_lgbm_tune,
                                              learning_rate = learning_rate,
                                              num_leaves = num_leaves,
                                              max_depth = max_depth,
                                              feature_fraction = feature_fraction,
                                              force_col_wise = FALSE,
                                              metric  = metric,
                                              nrounds = nrounds,
                                              workers = workers,
                                              num_class = n_class),
                 predict_fun = multiclass_lgbm_predict,
                 predict_tuned_fun = multiclass_lgbm_predict_tuned)
  }
)

# Methods -----------------------------------------------------------------
multiclass_lgbm_tune = function(features, tgt, wt = rep(1, nrow(features)),
                                tune_folds, learning_rate, num_leaves, max_depth,
                                feature_fraction, force_col_wise,
                                metric, nrounds, workers, num_class) {

  if (is.factor(tgt)) tgt = as.integer(tgt) - 1L
  if (min(tgt) != 0)  tgt = tgt - min(tgt)

  multiclass_lgbm_design = expand.grid(num_leaves = num_leaves,
                                       max_depth  = max_depth,
                                       learning_rate = learning_rate,
                                       feature_fraction = feature_fraction)

  multiclass_lgbm_tuning = purrr::pmap(multiclass_lgbm_design,
                                       function(...) {
                                         params = list(...)
                                         params$num_threads    = workers
                                         params$objective      = 'multiclass'
                                         params$force_col_wise = force_col_wise
                                         params$metric         = metric
                                         params$num_class      = num_class

                                         quiet_cv = purrr::quietly(lightgbm::lgb.cv)

                                         out = quiet_cv(params = params,
                                                        data   = features,
                                                        label  = tgt,
                                                        weight = wt,
                                                        folds  = split(seq_along(tune_folds), tune_folds),
                                                        nrounds = nrounds,
                                                        early_stopping_rounds = 20L,
                                                        verbose = -1)

                                         out$result
                                       })

  multiclass_lgbm_tuning_df = purrr::map(multiclass_lgbm_tuning, tidy_lgbm_cv)
  multiclass_lgbm_tuning_df = purrr::list_rbind(multiclass_lgbm_tuning_df,
                                                names_to = "idx_design")

  best_param_row = which.min(multiclass_lgbm_tuning_df$rmse)
  best_fit_iter  = multiclass_lgbm_tuning_df$idx_design[best_param_row]
  best_fit = multiclass_lgbm_tuning[[best_fit_iter]]

  # Get tuned predictions
  tgt_hat_list = purrr::map2(best_fit$boosters,
                             split(seq_along(tune_folds), tune_folds),
                             function(b, f) {
                               tgt_hat = stats::predict(b$booster, features[f,,drop=FALSE])
                               cbind(f, tgt_hat)
                             })

  tgt_hat_mtx = Reduce(rbind, tgt_hat_list)
  tgt_hat_mtx = tgt_hat_mtx[order(tgt_hat_mtx[,1]), ]

  # Return
  tune_res = list(
    tuned_model = tgt_hat_mtx[,-1],
    train_fun   = purrr::partial(multiclass_lgbm_train,
                                 params  = multiclass_lgbm_tuning_df$params[[best_param_row]],
                                 nrounds = multiclass_lgbm_tuning_df$nrounds[[best_param_row]]))

}

multiclass_lgbm_train = function(features, tgt, wt, params, nrounds) {

  if (is.factor(tgt)) tgt = as.integer(tgt) - 1L
  if (min(tgt) != 0)  tgt = tgt - min(tgt)

  quiet_lgb = purrr::quietly(lightgbm::lightgbm)
  fit = quiet_lgb(data = features,
                  label = tgt,
                  weight = wt,
                  params = params,
                  nrounds = nrounds,
                  verbose = -1)

  list(fit    = fit$result,
       params = c(params, nrounds = nrounds))
}

multiclass_lgbm_predict = function(model, features) {
  as.numeric(stats::predict(model$fit, features))
}

multiclass_lgbm_predict_tuned = function(model, features, tune_folds) {
  return(model)
}

