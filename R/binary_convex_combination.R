#' Convex Combination for Binary Classification
#' @export
#'
#' @return A model that fits a weighted average of the input probabilities,
#' with weights all nonnegative and summing to 1. Any number of input columns
#' is allowed.
#'
#' @details Inputs should be predicted probabilities in [0, 1]. Predictions are
#' the convex combination of the inputs, with weights chosen to maximize the
#' weighted Bernoulli log-likelihood (fitted probabilities are clamped away
#' from 0 and 1 inside the loss). With at most 3 input columns, the weights
#' are chosen by exhaustive search over a simplex grid with 100 steps. With
#' more columns, they are found by numerical optimization over
#' softmax-parametrized weights. Uses observation weights.
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

  features <- as.matrix(features)
  if (is.null(wt)) wt <- rep(1, length(tgt))
  if (is.factor(tgt)) tgt <- as.integer(tgt) - 1L

  eps <- 1e-15

  loss_fun <- function(fitted_vals) {
    -apply(pmin(pmax(fitted_vals, eps), 1 - eps), 2, bern_loglik,
           tgt = tgt, wt = wt)
  }

  convex_combo_weights(features, loss_fun, n_steps)
}

binary_convex_combo_predict <- function(model, features) {
  # inputs should be probabilities; clamp guards against out-of-range inputs
  pmin(pmax(as.numeric(as.matrix(features) %*% model), 0), 1)
}
