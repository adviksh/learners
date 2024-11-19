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
binary_lgbm = structure(
  class = c("learner_constructor", "function"),
  function(learning_rate = 0.05,
           num_leaves = 31,
           max_depth  = -1,
           feature_fraction = 1,
           force_col_wise = FALSE,
           metric  = 'rmse',
           nrounds = 10000L,
           workers = 1) {
    make_learner(name       = "binary_lgbm",
                 tune_fun    = purrr::partial(binary_lgbm_tune,
                                              learning_rate = learning_rate,
                                              num_leaves = num_leaves,
                                              max_depth = max_depth,
                                              feature_fraction = feature_fraction,
                                              force_col_wise = FALSE,
                                              metric  = metric,
                                              nrounds = nrounds,
                                              workers = workers),
                 predict_fun = binary_lgbm_predict,
                 predict_tuned_fun = binary_lgbm_predict_tuned)
  }
)

# Methods -----------------------------------------------------------------
binary_lgbm_tune = function(features, tgt, wt = rep(1, nrow(features)),
                            tune_folds, learning_rate, num_leaves, max_depth,
                            feature_fraction, force_col_wise,
                            metric, nrounds, workers) {

  binary_lgbm_design = expand.grid(num_leaves = num_leaves,
                                   max_depth  = max_depth,
                                   learning_rate = learning_rate,
                                   feature_fraction = feature_fraction)

  binary_lgbm_design = binary_lgbm_design[
    (binary_lgbm_design$max_depth <= 0) |
      (binary_lgbm_design$num_leaves <= 2^binary_lgbm_design$max_depth - 1)
      ,
    ]

  binary_lgbm_tuning = purrr::pmap(binary_lgbm_design,
                                   function(...) {
                                     params = list(...)
                                     params$num_threads    = workers
                                     params$objective      = 'binary'
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

  binary_lgbm_tuning_df = purrr::map(binary_lgbm_tuning, tidy_lgbm_cv)
  binary_lgbm_tuning_df = purrr::list_rbind(binary_lgbm_tuning_df,
                                            names_to = "idx_design")

  best_param_row = which.min(binary_lgbm_tuning_df$metric)
  best_fit_iter  = binary_lgbm_tuning_df$idx_design[best_param_row]
  best_fit = binary_lgbm_tuning[[best_fit_iter]]

  # Get tuned predictions
  tgt_hat_list = purrr::map2(best_fit$boosters,
                             split(seq_along(tune_folds), tune_folds),
                             function(b, f) {
                               data.frame(idx     = f,
                                          tgt_hat = predict(b$booster, features[f,,drop=FALSE]))
                             })

  tgt_hat_tb = purrr::list_rbind(tgt_hat_list)
  tgt_hat_tb = tgt_hat_tb[order(tgt_hat_tb$idx), ]

  # Return
  tune_res = list(
    tuned_model = as.numeric(tgt_hat_tb$tgt_hat),
    train_fun   = purrr::partial(binary_lgbm_train,
                                 params  = binary_lgbm_tuning_df$params[[best_param_row]],
                                 nrounds = binary_lgbm_tuning_df$nrounds[[best_param_row]]))

}

binary_lgbm_train = function(features, tgt, wt, params, nrounds) {

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

binary_lgbm_predict = function(model, features) {
  as.numeric(stats::predict(model$fit, features))
}

binary_lgbm_predict_tuned = function(model, features, tune_folds) {
  return(model)
}

