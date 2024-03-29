#' Logistic Regression
#' @export
#'
#' @return A logistic regression model for binary outcomes,
#' with class 'learner'.
binary_logistic <- structure(
  class = c("learner_constructor", "function"),
  function(engine = stats::glm.fit) {
    make_learner(
      name = "binary_logistic",
      train_fun = purrr::partial(binary_logistic_train, engine = engine),
      predict_fun = function(model, features) {
        z <- as.numeric(model[1] + (features %*% model[-1]))
        stats::plogis(z)
      }
    )
  }
)

binary_logistic_train = function(features, tgt, wt = NULL, engine) {

  if (is.null(wt)) wt = rep(1, length(tgt))
  if (is.factor(tgt)) tgt <- as.integer(tgt) - 1L

  tgt <- as.matrix(tgt)

  model <- engine(x = cbind(1, features),
                  y = tgt,
                  weights = wt,
                  family = stats::binomial())

  model_coef <- stats::coef(model)
  model_coef[is.na(model_coef)] <- 0

  return(model_coef)
}
