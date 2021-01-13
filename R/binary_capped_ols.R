#' Capped OLS Regression
#' @export
#'
#' @return A linear regression model for binary outcomes,
#' with class 'learner'. Predictions are capped to fall inside [0,1].
binary_capped_ols <- structure(
  class = c("learner_constructor", "function"),
  function() {
    make_learner(
      name = "regr_ols",
      train_fun = function(features, tgt, wt) {

        if (is.factor(tgt)) tgt <- as.integer(tgt) - 1L

        tgt <- as.matrix(tgt)

        model <- stats::glm.fit(x = cbind(1, features),
                                y = tgt,
                                weights = wt,
                                family = stats::gaussian())

        model_coef <- stats::coef(model)
        model_coef[is.na(model_coef)] <- 0

        return(model_coef)
      },
      predict_fun = function(model, features) {
        p <- as.numeric(model[1] + (features %*% model[-1]))
        pmax(pmin(p, 1), 0)
      }
    )
  }
)
