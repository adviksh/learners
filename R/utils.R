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

  data.frame(params  = I(list(result$params)),
             nrounds = result$best_ntreelimit,
             rmse    = result$evaluation_log$test_rmse_mean[result$best_iteration])
}

tidy_lgbm_cv = function(result) {

  params = result$boosters[[1]]$booster$params
  params$metric = params$metric[[1]]
  params$early_stopping_round = NULL

  data.frame(params  = I(list(params)),
             nrounds = result$best_iter,
             rmse    = result$best_score)
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
