#' OLS Regression
#' @export
#'
#' @return A linear regression model for continuous outcomes,
#' with class 'learner'.
regr_ols = structure(
  class = c("learner_constructor", "function"),
  function(engine = stats::glm.fit) {
    make_learner(
      name = "regr_ols",
      train_fun = purrr::partial(regr_ols_train, engine = engine),
      predict_fun = function(model, features) {
        as.numeric(model[1] + (features %*% model[-1]))
      }
    )
  }
)

regr_ols_train = function(features, tgt, wt = NULL, engine) {

  if (is.null(wt)) wt = rep(1, length(tgt))
  if (is.factor(tgt)) tgt = as.integer(tgt) - 1L

  tgt = as.matrix(tgt)

  model = engine(x = cbind(1, features),
                 y = tgt,
                 weights = wt,
                 family = stats::gaussian())

  model_coef = stats::coef(model)
  model_coef[is.na(model_coef)] = 0

  return(model_coef)
}
