context("continuous stacking learner")

test_suite_regr(regr_stacking)

test_that("regr stacking works with cheap learners", {
  learners <- list(ols = regr_ols(), samp = regr_sampmean())
  fit <- tune(regr_stacking(learners), toy_data$features, toy_data$tgt_continuous,
              toy_data$wt, toy_data$folds)
  fit <- train(fit, toy_data$features, toy_data$tgt_continuous, toy_data$wt)
  expect_length(predict(fit, toy_data$features), 100)
})
