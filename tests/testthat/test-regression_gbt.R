context("continuous GBT learner")

test_suite_regr(regr_gbt)

test_that("regr_gbt tune returns OOF predictions and trains/predicts", {
  set.seed(1)
  x <- toy_data$features
  y <- x[, 1] + rnorm(100, sd = .1)
  fit <- tune(regr_gbt(max_depth = 1:2), x, y, toy_data$wt, toy_data$folds)
  expect_length(fit$tuned_model, 100)
  fit <- train(fit, x, y, toy_data$wt)
  expect_length(predict(fit, x), 100)
  expect_length(predict_oos(regr_gbt(max_depth = 1:2), x, y, toy_data$wt,
                            toy_data$folds), 100)
})
