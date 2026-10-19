context("binary stacking learner")

test_suite_binary(binary_stacking)

test_that("binary stacking works with cheap learners and returns probabilities", {
  learners <- list(logit = binary_logistic(), samp = binary_sampmean())
  fit <- tune(binary_stacking(learners), toy_data$features, toy_data$tgt_binary,
              toy_data$wt, toy_data$folds)
  fit <- train(fit, toy_data$features, toy_data$tgt_binary, toy_data$wt)
  pred <- predict(fit, toy_data$features)
  expect_length(pred, 100)
  expect_true(all(pred >= 0 & pred <= 1))

  # single component
  fit1 <- tune(binary_stacking(list(logit = binary_logistic())),
               toy_data$features, toy_data$tgt_binary, toy_data$wt, toy_data$folds)
  fit1 <- train(fit1, toy_data$features, toy_data$tgt_binary, toy_data$wt)
  expect_length(predict(fit1, toy_data$features), 100)
})
