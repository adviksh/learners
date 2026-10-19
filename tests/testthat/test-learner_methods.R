context("test-learner_methods")

test_that("No tuning for NULL tune_fun or train_fun", {
  lrn <- make_learner()

  expect_identical(lrn,
                   tune(object = lrn,
                        features = toy_data$x,
                        tgt = toy_data$tgt_binary,
                        wt = toy_data$wt,
                        tune_folds = toy_data$folds))
  expect_identical(lrn,
                   train(object = lrn,
                         features = toy_data$x,
                         tgt = toy_data$tgt_binary,
                         wt = toy_data$wt))
})

# tune_predict_oos_fold ---------------------------------------------------
test_that("tune_predict_oos_fold maintains out-of-sample status", {

  detect_predict <- function(model, features) {

    if (!identical(features, matrix(1L))) {
      stop("OOS fold is not sole target for prediction")
    }

  }

  detect_train <- function(features, tgt, wt) {
    if (1 %in% tgt) stop("OOS fold caught in training")
  }

  detect_tune <- function(features, tgt, wt, tune_folds) {
    if (1 %in% tgt) stop("OOS fold caught in tuning")

    list(train_fun = detect_train,
         tuned_model = NULL)
  }

  detecting_learner <- make_learner(tune_fun    = detect_tune,
                                    predict_fun = detect_predict)

  expect_error(tune_predict_oos_fold(detecting_learner,
                                     features = matrix(1:10, ncol = 1),
                                     tgt = 1:10,
                                     tune_folds = 1:10,
                                     which_fold = 1),
               NA)

})

# count the number of times each method gets called
test_that("tune_predict_oos ...", {
  tally_predict <- function(model, features) {
    predict_count[features] <<- predict_count[features] + 1
    return(-1)
  }

  tally_train <- function(features, tgt, wt) {
    train_count[features] <<- train_count[features] + 1
  }

  tally_tune <- function(features, tgt, wt, tune_folds) {

    tune_count[features] <<- tune_count[features] + 1

    list(train_fun  = tally_train,
         tuned_model = NULL)
  }

  counting_learner <- make_learner(
    tune_fun   = tally_tune,
    predict_fun = tally_predict
  )

  features <- matrix(1:20)
  folds <- rep_len(1:5, 20)
  tune_count    <- rep(0, length(features))
  train_count   <- rep(0, length(features))
  predict_count <- rep(0, length(features))

  pred <- tune_predict_oos(counting_learner,
                           features = features,
                           tgt = NULL,
                           tune_folds = folds)

  expect_identical(tune_count, rep(4, length(features)))
  expect_identical(train_count, rep(4, length(features)))
  expect_identical(predict_count, rep(1, length(features)))
})

# holdout (NA fold) handling -----------------------------------------------
test_that("predict_oos returns length n with NA at holdout rows (tuned_model)", {
  set.seed(1)
  n <- 100
  x <- matrix(rnorm(n * 5), nrow = n)
  y <- rep_len(0:1, n)
  folds <- rep_len(1:5, n)
  folds[c(3, 50, 99)] <- NA

  pred <- predict_oos(binary_elasticnet(), x, y, tune_folds = folds)

  expect_length(pred, n)
  expect_identical(is.na(pred), is.na(folds))
})

test_that("predict_oos returns length n with NA at holdout rows (no tuned_model)", {
  folds <- rep_len(1:5, 100)
  folds[c(3, 50, 99)] <- NA

  pred <- predict_oos(binary_random(), toy_data$features, toy_data$tgt_binary,
                      tune_folds = folds)

  expect_length(pred, 100)
  expect_identical(is.na(pred), is.na(folds))
})

test_that("predict_oos places tuned predictions at the right rows", {
  tune_fun <- function(features, tgt, wt, tune_folds) {
    list(train_fun = NULL, tuned_model = features[, 1] * 10)
  }
  lrn <- make_learner(tune_fun = tune_fun,
                      predict_fun = function(model, features) 0,
                      predict_tuned_fun = function(model, features, tune_folds) model)

  x <- matrix(1:10, ncol = 1)
  folds <- c(NA, 1, 2, 1, 2, NA, 1, 2, 1, 2)
  pred <- predict_oos(lrn, x, 1:10, tune_folds = folds)

  expect_identical(pred, ifelse(is.na(folds), NA, as.numeric(x) * 10))

  # matrix-valued tuned_model
  lrn$tune_fun <- function(features, tgt, wt, tune_folds) {
    list(train_fun = NULL, tuned_model = cbind(features[, 1], -features[, 1]))
  }
  pred <- predict_oos(lrn, x, 1:10, tune_folds = folds)
  expect_equal(dim(pred), c(10, 2))
  expect_identical(is.na(pred[, 1]), is.na(folds))
  expect_equal(pred[!is.na(folds), 1], as.numeric(x[!is.na(folds)]))
})

test_that("holdout rows never reach tune or train in oos methods", {
  features <- matrix(1:20, ncol = 1)
  folds <- rep_len(1:4, 20)
  holdout <- c(2, 7, 11)
  folds[holdout] <- NA
  tgt <- 1:20

  detect_train <- function(features, tgt, wt) {
    if (any(holdout %in% tgt)) stop("holdout caught in training")
    NULL
  }
  detect_tune <- function(features, tgt, wt, tune_folds) {
    if (any(holdout %in% tgt)) stop("holdout caught in tuning")
    if (anyNA(tune_folds)) stop("NA fold in tuning")
    list(train_fun = detect_train, tuned_model = NULL)
  }
  lrn <- make_learner(tune_fun = detect_tune,
                      predict_fun = function(model, features) rep(0, nrow(features)))

  expect_error(p1 <- predict_oos(lrn, features, tgt, tune_folds = folds), NA)
  expect_error(p2 <- tune_predict_oos(lrn, features, tgt, tune_folds = folds), NA)
  expect_identical(is.na(p1), is.na(folds))
  expect_identical(is.na(p2), is.na(folds))
})
