#' Random Noise for Classification
#' @export
#'
#' @param seed Random seed for replicability.
#'
#' @return A model that makes random predictions for a binary outcome
#'
binary_random <- structure(
  class = c("learner_constructor", "function"),
  function(seed = 1) {
    make_learner(name        = "binary_random",
                 tune_fun    = purrr::partial(binary_random_tune, seed = seed),
                 train_fun   = purrr::partial(binary_random_train, seed = seed),
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
  # draw reproducibly from the seed without disturbing the caller's RNG state
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    old_seed = get(".Random.seed", envir = globalenv(), inherits = FALSE)
    on.exit(assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
  } else {
    on.exit(if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      rm(".Random.seed", envir = globalenv())
    }, add = TRUE)
  }
  set.seed(model)
  stats::runif(nrow(features))
}
