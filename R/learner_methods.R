#' @rdname learner
#' @export
tune.learner <- function(object, features, tgt, wt = NULL, tune_folds, ...) {

  # Defense -----------------------------------------------------------------
  if (missing(object)) rlang::abort("object is missing")
  if (missing(features)) rlang::abort("features is missing")
  if (missing(tgt)) rlang::abort("tgt is missing")
  if (missing(wt)) rlang::abort("wt is missing")
  if (missing(tune_folds)) rlang::abort("tune_folds is missing")

  # Fast Return -------------------------------------------------------------
  if (is.null(object$tune_fun)) return(object)

  # Body --------------------------------------------------------------
  if (is.null(wt)) wt <- rep(1L, nrow(features))

  tune_idx <- which(!is.na(tune_folds))

  object$train_fun <- object$tune_fun(features   = features[tune_idx, , drop = FALSE],
                                      tgt        = tgt[tune_idx],
                                      wt         = wt[tune_idx],
                                      tune_folds = tune_folds[tune_idx])

  return(object)

}

#' @rdname learner
#' @export
train.learner <- function(object, features, tgt, wt = NULL, ...) {

  # Defense -----------------------------------------------------------------
  if (missing(object)) rlang::abort("object is missing")
  if (missing(features)) rlang::abort("features is missing")
  if (missing(tgt)) rlang::abort("tgt is missing")
  if (missing(wt)) rlang::abort("wt is missing")

  # Fast Return -------------------------------------------------------------
  if (is.null(object$train_fun)) return(object)

  # Body --------------------------------------------------------------
  if (is.null(wt)) wt <- rep(1L, nrow(features))

  object$model <- object$train_fun(features = features,
                                   tgt      = tgt,
                                   wt       = wt)

  return(object)
}

#' @rdname learner
#' @export
predict.learner <- function(object, newdata, ...) {

  object$predict_fun(object$model, newdata)

}

#' @rdname learner
#' @export
tune_predict_ins.learner <- function(object, features, tgt, wt = NULL,
                                     tune_folds, ...) {

  if (is.null(wt)) wt <- rep(1L, nrow(features))

  tuned_learner   <- tune(object, features, tgt, wt, tune_folds)
  trained_learner <- train(tuned_learner, features, tgt, wt)

  # Return
  stats::predict(trained_learner, features)

}

#' @rdname learner
#' @export
tune_predict_oos.learner <- function(object, features, tgt, wt = NULL,
                                     tune_folds, ...) {

  if (is.null(wt)) wt <- rep(1, nrow(features))

  fold_vals <- setdiff(unique(tune_folds), NA)

  pred_list <- lapply(fold_vals,
                      tune_predict_oos_fold,
                      object     = object,
                      features   = features,
                      tgt        = tgt,
                      wt         = wt,
                      tune_folds = tune_folds)

  predictions <- rep(NA, length(tgt))

  for (ii in seq_along(fold_vals)) {
    predictions[which(tune_folds == fold_vals[ii])] <- pred_list[[ii]]
  }

  return(predictions)

}

#' @rdname learner
#' @export
tune_predict_oos_fold.learner <- function(object, features, tgt, wt = NULL,
                                          tune_folds, which_fold, ...) {

  if (is.null(wt)) wt <- rep(1, nrow(features))

  tune_idx <- which(tune_folds != which_fold)
  pred_idx <- which(tune_folds == which_fold)

  tuned_learner <- tune(object = object,
                        features = features[tune_idx, , drop = FALSE],
                        tgt = tgt[tune_idx],
                        wt = wt[tune_idx],
                        tune_folds = tune_folds[tune_idx])

  trained_learner <- train(object = tuned_learner,
                           features = features[tune_idx, , drop = FALSE],
                           tgt = tgt[tune_idx],
                           wt = wt[tune_idx])

  predictions <- stats::predict(trained_learner,
                                features[pred_idx, , drop = FALSE])

  return(predictions)

}
