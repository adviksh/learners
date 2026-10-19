context("binary random learner")

test_suite_binary(binary_random)

test_that("train works without tune", {
  lrn <- train(binary_random(seed = 3), toy_data$features, toy_data$tgt_binary,
               toy_data$wt)
  expect_length(predict(lrn, toy_data$features), nrow(toy_data$features))
})
