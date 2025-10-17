#' Gradient Boosted Tree Regression
#' @export
#'
#' @description Construct a gradient boosted tree regression using lightgbm
#' All parameters can be passed as single values, or as vectors. Performs
#' grid search over all combinations of provided values, using early stopping
#' to select the best number of boosting iterations.
#'
#' @return A gradient boosted tree model for continuous outcomes,
#' with class 'learner'.
#'
#' @param learning_rate learning rate.
#' @param max_depth maximum depth of a tree.
#' @param feature_fraction bagging_fraction ratio of columns when constructing
#' each tree. Values less than 1 will speed up computation, and may reduce
#' overfitting.
#' @param workers number of threads to use, to speed up training and prediction
#'
regr_lgbm = structure(
  class = c("learner_constructor", "function"),
  function(learning_rate = 0.05,
           num_leaves = 31,
           max_depth  = -1,
           min_data_in_leaf = 20,
           feature_fraction = 1,
           force_col_wise = FALSE,
           metric  = 'l2',
           nrounds = 10000L,
           workers = 1) {
    make_learner(name       = "regr_lgbm",
                 tune_fun    = purrr::partial(regr_lgbm_tune,
                                              learning_rate = learning_rate,
                                              num_leaves = num_leaves,
                                              max_depth = max_depth,
                                              min_data_in_leaf = min_data_in_leaf,
                                              feature_fraction = feature_fraction,
                                              force_col_wise = FALSE,
                                              metric  = metric,
                                              nrounds = nrounds,
                                              workers = workers),
                 predict_fun = regr_lgbm_predict,
                 predict_tuned_fun = regr_lgbm_predict_tuned)
  }
)

# Methods -----------------------------------------------------------------
regr_lgbm_tune = function(features, tgt, wt = rep(1, nrow(features)),
                          tune_folds, learning_rate, num_leaves, max_depth,
                          min_data_in_leaf,
                          feature_fraction, force_col_wise,
                          metric, nrounds, workers) {

  regr_lgbm_design = expand.grid(num_leaves = num_leaves,
                                 max_depth  = max_depth,
                                 min_data_in_leaf = min_data_in_leaf,
                                 learning_rate    = learning_rate,
                                 feature_fraction = feature_fraction)

  # cap the number of leaves (2^regr_lgbm_design$max_depth - 1)
  regr_lgbm_design$num_leaves = ifelse(regr_lgbm_design$max_depth <= 0,
                                       regr_lgbm_design$num_leaves,
                                       pmin(regr_lgbm_design$num_leaves,
                                            2^regr_lgbm_design$max_depth - 1))

  regr_lgbm_tuning = purrr::pmap(regr_lgbm_design,
                                 function(...) {
                                   params = list(...)
                                   params$num_threads    = workers
                                   params$objective      = 'regression'
                                   params$force_col_wise = force_col_wise
                                   params$metric         = metric

                                   quiet_cv = purrr::quietly(lightgbm::lgb.cv)

                                   lgb_data = lightgbm::lgb.Dataset(data   = features,
                                                                    label  = tgt,
                                                                    weight = wt)

                                   out = quiet_cv(params = params,
                                                  data   = lgb_data,
                                                  folds  = split(seq_along(tune_folds), tune_folds),
                                                  nrounds = nrounds,
                                                  early_stopping_rounds = 20L,
                                                  verbose = -1)

                                   out$result
                                 })

  regr_lgbm_tuning_df = purrr::map(regr_lgbm_tuning, tidy_lgbm_cv)
  regr_lgbm_tuning_df = purrr::list_rbind(regr_lgbm_tuning_df,
                                          names_to = "idx_design")

  best_param_row = which.min(regr_lgbm_tuning_df$metric)
  best_fit_iter  = regr_lgbm_tuning_df$idx_design[best_param_row]
  best_fit = regr_lgbm_tuning[[best_fit_iter]]

  # Get tuned predictions
  tgt_hat_list = purrr::map2(best_fit$boosters,
                             split(seq_along(tune_folds), tune_folds),
                             function(b, f) {
                               data.frame(idx     = f,
                                          tgt_hat = stats::predict(b$booster, features[f,,drop=FALSE]))
                             })

  tgt_hat_tb = purrr::list_rbind(tgt_hat_list)
  tgt_hat_tb = tgt_hat_tb[order(tgt_hat_tb$idx), ]

  # Return
  tune_res = list(
    tuned_model = as.numeric(tgt_hat_tb$tgt_hat),
    train_fun   = purrr::partial(regr_lgbm_train,
                                 params  = regr_lgbm_tuning_df$params[[best_param_row]],
                                 nrounds = regr_lgbm_tuning_df$nrounds[[best_param_row]]))

}

regr_lgbm_train = function(features, tgt, wt, params, nrounds) {

  if (is.factor(tgt)) tgt = as.integer(tgt) - 1L

  quiet_lgb = purrr::quietly(lightgbm::lightgbm)
  fit = quiet_lgb(data = features, label = tgt,
                  weight = wt,
                  params = params,
                  nrounds = nrounds,
                  verbose = -1)

  list(fit    = fit$result,
       params = c(params, nrounds = nrounds))
}

regr_lgbm_predict = function(model, features) {
  as.numeric(stats::predict(model$fit, features))
}

regr_lgbm_predict_tuned = function(model, features, tune_folds) {
  return(model)
}

