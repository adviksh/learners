# Suite -------------------------------------------------------------------
test_suite_regr <- function(constructor) {
  test_that("construction", { test_construct_regr(constructor) })

  test_that("tuning", { test_tune_regr(constructor) })

  test_that("training", { test_train_regr(constructor) })

  test_that("prediction", { test_predict_regr(constructor) })

  test_that("ins prediction", { test_tune_predict_ins_regr(constructor) })

  test_that("oos prediction", { test_tune_predict_oos_regr(constructor) })
}

# Helpers -----------------------------------------------------------------
test_construct_regr <- function(constructor, ...) {
  expect_error(constructor(...), NA)
}

test_tune_regr <- function(constructor, ...) {
  base_learner <- constructor(...)

  expect_error(tune(base_learner,
                    features   = toy_data$features,
                    tgt        = toy_data$tgt_continuous,
                    wt         = toy_data$wt,
                    tune_folds = toy_data$folds),
               NA)
}

test_train_regr <- function(constructor, ...) {
  base_learner <- constructor(...)

  tuned_learner <- tune(base_learner,
                        features   = toy_data$features,
                        tgt        = toy_data$tgt_continuous,
                        wt         = toy_data$wt,
                        tune_folds = toy_data$folds)

  expect_error(train(tuned_learner,
                     features = toy_data$features,
                     tgt      = toy_data$tgt_continuous,
                     wt       = toy_data$wt),
               NA)
}

test_predict_regr <- function(constructor, ...) {
  base_learner <- constructor(...)

  tuned_learner <- tune(base_learner,
                        features   = toy_data$features,
                        tgt        = toy_data$tgt_continuous,
                        wt         = toy_data$wt,
                        tune_folds = toy_data$folds)

  trained_learner <- train(tuned_learner,
                           features = toy_data$features,
                           tgt      = toy_data$tgt_continuous,
                           wt       = toy_data$wt)

  expect_error(stats::predict(trained_learner,
                              toy_data$features),
               NA)
}

test_tune_predict_ins_regr <- function(constructor, ...) {
  base_learner <- constructor(...)

  expect_error(tune_predict_ins(base_learner,
                                features = toy_data$features,
                                tgt      = toy_data$tgt_continuous,
                                wt       = toy_data$wt,
                                tune_folds = toy_data$folds),
               NA)
}

test_tune_predict_oos_regr <- function(constructor, ...) {
  base_learner <- constructor(...)

  expect_error(tune_predict_oos(base_learner,
                                features = toy_data$features,
                                tgt      = toy_data$tgt_continuous,
                                wt       = toy_data$wt,
                                tune_folds = toy_data$folds),
               NA)
}
