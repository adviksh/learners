#' Convex Combination for Regression
#' @export
#'
#' @return A model that fits a weighted average of the inputs, with weights
#' all nonnegative and summing to 1. Any number of input columns is allowed.
#'
#' @details Minimizes the weighted sum of squared errors, using the
#' observation weights \code{wt}. With at most 3 input columns, the weights are
#' chosen by exhaustive search over a simplex grid with 100 steps. With more
#' columns the grid grows combinatorially, so the weights are instead found
#' by numerical optimization over softmax-parametrized weights.
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

  features <- as.matrix(features)
  if (is.null(wt)) wt <- rep(1, length(tgt))

  loss_fun <- function(fitted_vals) {
    colSums(wt * (fitted_vals - tgt)^2)
  }

  convex_combo_weights(features, loss_fun, n_steps)
}

regr_convex_combo_predict <- function(model, features) {
  as.numeric(as.matrix(features) %*% model)
}

# Helpers -----------------------------------------------------------------
# Finds simplex weights minimizing loss_fun, which maps a matrix of candidate
# fitted values (one column per candidate) to a vector of losses.
# Uses a grid search when the grid is small (<= 3 columns); otherwise
# optimizes over softmax-parametrized weights.
convex_combo_weights <- function(features, loss_fun, n_steps) {

  k <- ncol(features)

  if (k == 1) return(stats::setNames(1, colnames(features)))

  if (k <= 3) {
    possible_coefs <- simplex_grid(k, n_steps)
    losses         <- loss_fun(features %*% possible_coefs)
    best_coef      <- as.numeric(possible_coefs[, which.min(losses)])
  } else {
    softmax <- function(z) { e <- exp(z - max(z)); e / sum(e) }
    obj <- function(z) {
      as.numeric(loss_fun(features %*% as.matrix(softmax(z))))
    }
    opt       <- stats::optim(rep(0, k), obj, method = "BFGS")
    best_coef <- softmax(opt$par)
    # also consider each single input, so a corner solution is never missed
    corners   <- diag(k)
    corner_l  <- loss_fun(features %*% corners)
    if (min(corner_l) < obj(opt$par)) best_coef <- corners[, which.min(corner_l)]
  }

  stats::setNames(best_coef, colnames(features))
}
