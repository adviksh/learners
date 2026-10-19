context("continuous elasticnet learner")

test_suite_regr(regr_elasticnet)

test_that("future plan is restored after tuning with workers", {
  skip_on_os("windows")
  old <- future::plan()
  suppressWarnings(tune(regr_elasticnet(workers = 2), toy_data$features, toy_data$tgt_continuous,
       toy_data$wt, toy_data$folds))
  expect_identical(class(future::plan()), class(old))
})
