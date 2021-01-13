#' Sample Mean for Classification
#' @export
#'
#' @return A model that predicts the weighted sample mean
#'
binary_sampmean <- structure(
  class = c("learner_constructor", "function"),
  function() {
    make_learner(name        = "binary_sampmean",
                 tune_fun    = NULL,
                 train_fun   = sampmean_train,
                 predict_fun = sampmean_predict)
  }
)

# Methods -----------------------------------------------------------------
sampmean_train <- function(features, tgt, wt, params) {

  if (is.null(wt)) wt <- rep(1L, length(tgt))
  if (is.factor(tgt)) tgt <- as.integer(tgt) - 1L

  stats::weighted.mean(tgt, wt)
}

sampmean_predict <- function(model, features) { rep(model, nrow(features)) }
