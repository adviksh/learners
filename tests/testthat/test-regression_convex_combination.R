context("regression convex combination learner")

test_suite_regr(regr_convex_combo)

test_that("regr convex combo handles 1, 3, and 5 columns", {
  set.seed(1)
  n <- 300
  y <- rnorm(n)
  noise <- function() rnorm(n, sd = 3)

  f1 <- train(regr_convex_combo(), cbind(a = y), y)
  expect_equal(unname(predict(f1, cbind(a = y))), y)

  x3 <- cbind(n1 = noise(), a = y + rnorm(n, sd = .1), n2 = noise())
  f3 <- train(regr_convex_combo(), x3, y)
  expect_length(f3$trained_model, 3)
  expect_gt(f3$trained_model[["a"]], 0.8)
  expect_length(predict(f3, x3), n)

  x5 <- cbind(noise(), noise(), noise(), noise(), a = y + rnorm(n, sd = .1))
  colnames(x5)[1:4] <- paste0("n", 1:4)
  f5 <- train(regr_convex_combo(), x5, y)
  expect_equal(sum(f5$trained_model), 1)
  expect_gt(f5$trained_model[["a"]], 0.7)
  expect_length(predict(f5, x5), n)
})

test_that("regr convex combo respects weights", {
  set.seed(2)
  n <- 200
  y <- rnorm(n)
  x <- cbind(a = y, b = y + 5)
  # a is right on the first half, b is right on the second half
  x[101:200, ] <- cbind(y[101:200] + 5, y[101:200])
  wt_a <- c(rep(1, 100), rep(0, 100))

  fa <- train(regr_convex_combo(), x, y, wt = wt_a)
  fb <- train(regr_convex_combo(), x, y, wt = 1 - wt_a)
  expect_equal(fa$trained_model[["a"]], 1)
  expect_equal(fb$trained_model[["b"]], 1)
})
