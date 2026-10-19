simplex_grid <- function(n_points, n_steps) {

  integer_grid(n_points, n_steps) / n_steps

}

integer_grid <- function(n_points, n_steps) {

  if (n_points == 1) return(n_steps)

  # call recursively
  grids <- purrr::map(0:n_steps,
                      ~rbind(.x,
                             integer_grid(n_points = n_points - 1,
                                          n_steps = n_steps - .x)))
  Reduce(cbind, grids)

}

tidy_xgb_cv <- function(result){

  # best_iteration is the number of rounds in xgboost >= 2 (best_ntreelimit was removed)
  nrounds = result$best_iteration
  if (is.null(nrounds)) nrounds = result$best_ntreelimit

  data.frame(params  = I(list(result$params)),
             nrounds = nrounds,
             rmse    = result$evaluation_log$test_rmse_mean[nrounds])
}

tidy_lgbm_cv = function(result) {

  params = result$boosters[[1]]$booster$params
  params$metric = params$metric[[1]]

  # lgb.cv stores the number of rounds (and early stopping) in params; these would
  # override the tuned `nrounds` passed to lightgbm() at train time, so strip them
  drop_params = c("early_stopping_round", "early_stopping_rounds", "early_stopping",
                  "n_iter_no_change",
                  "num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees",
                  "num_round", "num_rounds", "nrounds", "num_boost_round",
                  "n_estimators", "max_iter")
  params = params[setdiff(names(params), drop_params)]

  data.frame(params  = I(list(params)),
             nrounds = result$best_iter,
             metric  = result$best_score)
}

# Index of the best row, given a LightGBM metric name and the CV scores.
# Higher-is-better metrics (auc, ndcg, map, ...) are maximized, others minimized.
lgbm_best_row = function(metric, scores) {

  higher_better = c("auc", "auc_mu", "average_precision", "map",
                    "mean_average_precision", "ndcg", "lambdarank",
                    "rank_xendcg", "xendcg", "xe_ndcg", "xe_ndcg_mart", "xendcg_mart")

  metric = sub("@.*$", "", metric[[1]])

  if (metric %in% higher_better) which.max(scores) else which.min(scores)
}

bern_loglik <- function(tgt_hat, tgt, wt) {
  sum(wt * stats::dbinom(tgt, size = 1, prob = tgt_hat, log = TRUE))
}

#' Check if Object is a Learner
#' @export
#'
#' @param x object to check
is_learner <- function(x) { inherits(x, "learner") }

#' Check if Object is a Learner Constructor
#' @export
#'
#' @param x object to check
is_learner_constructor <- function(x) { inherits(x, "learner_constructor") }
