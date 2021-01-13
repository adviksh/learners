#' Convex Combination for Regression
#' @export
#'
#' @return A model that fits a weighted average of the inputs, with weights
#' all nonnegative and summing to 1.
#'
#' @details Does not use weights.
#'
binary_convex_combo <- structure(
  class = c("learner_constructor", "function"),
  function() {
    make_learner(name        = "binary_convex_combo",
                 tune_fun    = NULL,
                 train_fun   = purrr::partial(binary_convex_combo_train,
                                              n_steps = 100),
                 predict_fun = binary_convex_combo_predict)
  }
)

# Methods -----------------------------------------------------------------
binary_convex_combo_train <- function(features, tgt, wt, n_steps) {

  if (ncol(features) == 1) return(1)

  if (ncol(features) > 2) features <- features[,1:2]

  possible_coefs <- simplex_grid(ncol(features), n_steps)

  fitted_vals <- stats::plogis(features %*% possible_coefs)
  loglik      <- apply(fitted_vals, 2, bern_loglik,
                       tgt = tgt,
                       wt = wt)
  best_fit    <- which.max(loglik)
  best_coef   <- as.numeric(possible_coefs[,best_fit])

  stats::setNames(best_coef, colnames(features))

}

binary_convex_combo_predict <- function(model, features) {
  z <- as.numeric(features[,1:2] %*% model)
  stats::plogis(z)
}


# Helpers -----------------------------------------------------------------
