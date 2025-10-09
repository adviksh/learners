# Suite -------------------------------------------------------------------
test_suite_binary <- function(constructor) {
  test_that("construction", { test_construct_binary(constructor) })

  test_that("tuning", { test_tune_binary(constructor) })

  test_that("training", { test_train_binary(constructor) })

  test_that("prediction", { test_predict_binary(constructor) })

  test_that("ins prediction", { test_tune_predict_ins_binary(constructor) })

  test_that("oos prediction", { test_tune_predict_oos_binary(constructor) })
}

# Helpers -----------------------------------------------------------------
test_construct_binary <- function(constructor, ...) {
  expect_error(constructor(...), NA)
}

test_tune_binary <- function(constructor,
                             features = toy_data$features,
                             tgt      = toy_data$tgt_binary,
                             wt       = toy_data$wt,
                             tune_folds = toy_data$folds,
                             ...) {
  base_learner <- constructor(...)

  expect_error(tune(base_learner,
                    features = features,
                    tgt      = tgt,
                    wt       = wt,
                    tune_folds = tune_folds),
               NA)
  
  expect_error(tune(base_learner,
                    features = features,
                    tgt      = tgt,
                    tune_folds = tune_folds),
               NA)
}

test_train_binary <- function(constructor,
                              features = toy_data$features,
                              tgt      = toy_data$tgt_binary,
                              wt       = toy_data$wt,
                              tune_folds = toy_data$folds,
                              ...) {
  base_learner <- constructor(...)

  tuned_learner <- tune(base_learner,
                        features   = features,
                        tgt        = tgt,
                        wt         = wt,
                        tune_folds = tune_folds)

  expect_error(train(tuned_learner,
                     features   = features,
                     tgt        = tgt,
                     wt         = wt),
               NA)
}

test_predict_binary <- function(constructor,
                                features = toy_data$features,
                                tgt      = toy_data$tgt_binary,
                                wt       = toy_data$wt,
                                tune_folds = toy_data$folds,
                                ...) {
  base_learner <- constructor(...)

  tuned_learner <- tune(base_learner,
                        features   = features,
                        tgt        = tgt,
                        wt         = wt,
                        tune_folds = tune_folds)

  trained_learner <- train(tuned_learner,
                           features = features,
                           tgt      = tgt,
                           wt       = wt)

  predictions <- stats::predict(trained_learner,
                                features)

  expect_true(any(is.na(predictions)) == FALSE)
  expect_true(min(predictions) >= 0)
  expect_true(max(predictions) <= 1)
}

test_tune_predict_ins_binary <- function(constructor, ...) {
  base_learner <- constructor(...)

  expect_error(tune_predict_ins(base_learner,
                                features = toy_data$features,
                                tgt      = toy_data$tgt_binary,
                                wt       = toy_data$wt,
                                tune_folds = toy_data$folds),
               NA)
}

test_tune_predict_oos_binary <- function(constructor,
                                         features = toy_data$features,
                                         tgt      = toy_data$tgt_binary,
                                         wt       = toy_data$wt,
                                         tune_folds = toy_data$folds,
                                         ...) {
  base_learner <- constructor(...)

  predictions <- tune_predict_oos(base_learner,
                                  features   = features,
                                  tgt        = tgt,
                                  wt         = wt,
                                  tune_folds = tune_folds)

  expect_true(any(is.na(predictions)) == FALSE)
  expect_true(min(predictions) >= 0)
  expect_true(max(predictions) <= 1)
}
