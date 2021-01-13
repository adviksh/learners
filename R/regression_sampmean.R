#' Sample Mean for Regression
#' @export
#'
#' @return A model that predicts the weighted sample mean
#'
regr_sampmean <- structure(
  class = c("learner_constructor", "function"),
  function() {
    make_learner(name       = "regr_sampmean",
                 tune_fun    = NULL,
                 train_fun   = sampmean_train,
                 predict_fun = sampmean_predict)
  }
)

# Methods -----------------------------------------------------------------
sampmean_train <- function(features, tgt, wt, params) {

  if (is.null(wt)) wt <- rep(1L, length(tgt))

  stats::weighted.mean(tgt, wt)
}

sampmean_predict <- function(model, features) { rep(model, nrow(features)) }
