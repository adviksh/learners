#' Convex Combination for Regression
#'
#' @return A model that fits a weighted average of the inputs, with weights
#' all nonnegative and summing to 1.
#'
#' @details Does not use weights.
#'
regr_convex_combo <- structure(
  class = c("learner_constructor", "function"),
  function() {
    make_learner(name       = "regr_convex_combo",
                 tune_fun    = NULL,
                 train_fun   = purrr::partial(regr_convex_combo_train,
                                              n_steps = 100),
                 predict_fun = regr_convex_combo_predict)
  }
)

# Methods -----------------------------------------------------------------
regr_convex_combo_train <- function(features, tgt, wt, n_steps) {

  if (ncol(features) == 1) return(1)

  if (ncol(features) > 2) features <- features[,1:2]

  possible_coefs <- simplex_grid(ncol(features), n_steps)

  fitted_vals <- features %*% possible_coefs
  sse         <- colSums((fitted_vals - tgt)^2)
  best_fit    <- which.min(sse)
  best_coef   <- as.numeric(possible_coefs[,best_fit])

  stats::setNames(best_coef, colnames(features))

}

regr_convex_combo_predict <- function(model, features) {
  as.numeric(features[,1:2] %*% model)
}
