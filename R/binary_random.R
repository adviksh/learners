#' Random Noise for Classification
#' @export
#'
#' @return A model that makes random predictions for a binary outcome
#'
binary_random <- structure(
  class = c("learner_constructor", "function"),
  function(seed = 1) {
    make_learner(name        = "binary_sampmean",
                 tune_fun    = purrr::partial(binary_random_tune, seed = seed),
                 train_fun   = binary_random_train,
                 predict_fun = binary_random_predict,
                 predict_tuned_fun = NULL)
  }
)

# Methods -----------------------------------------------------------------
binary_random_tune = function(features, tgt, wt, tune_folds, seed) {

  tune_res = list(
    tuned_model = NULL,
    train_fun   = purrr::partial(binary_random_train, seed = seed)
  )

  tune_res
}

binary_random_train <- function(features, tgt, wt, seed) {
  return(seed)
}

binary_random_predict <- function(model, features) {
  set.seed(model)
  runif(nrow(features))
}
