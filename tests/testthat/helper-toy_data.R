toy_data <- list(
  features       = matrix(rnorm(1000), nrow = 100),
  tgt_binary     = rep_len(0:1, 100),
  tgt_continuous = rnorm(100),
  wt             = rep(1, 100),
  folds          = rep_len(1:5, 100)
)
