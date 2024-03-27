#' Methods
#' @export
#' @rdname methods
#'
#' @param object a model
#' @param features input data, where each observation is a row
#' @param tgt the variable to predict
#' @param wt optional, weights for each observation
#' @param tune_folds integer vector, a division of data into cross-validation folds
#' @param which_fold for tune_predict_oos_fold, which fold to retrun out of sample
#' predictions for.
#' @param ... additional arguments
tune <- function(object, features, tgt, wt = NULL, tune_folds, ...) {
  UseMethod("tune")
}

#' @export
#' @rdname methods
train <- function(object, features, tgt, wt = NULL, ...) {
  UseMethod("train")
}

#' @export
#' @rdname methods
tune_predict_ins <- function(object, features, tgt, wt = NULL, tune_folds,
                             ...) {
  UseMethod("tune_predict_oos")
}

#' @export
#' @rdname methods
predict_oos <- function(object, features, tgt, wt = NULL, tune_folds,
                        ...) {
  UseMethod("tune_predict_oos")
}
#' @export
#' @rdname methods
tune_predict_oos <- function(object, features, tgt, wt = NULL, tune_folds,
                             ...) {
  UseMethod("tune_predict_oos")
}

#' @export
#' @rdname methods
tune_predict_oos_fold <- function(object, features, tgt, wt = NULL, tune_folds,
                                  which_fold, ...) {
  UseMethod("tune_predict_oos_fold")
}
