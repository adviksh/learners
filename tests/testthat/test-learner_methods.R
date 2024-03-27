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
