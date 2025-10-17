#' Elasticnet, then Boosted Trees
#' @export
#' 
#' @description Logistic elasticnet regression, with gradient boosted trees to pick the residual
#'
#' @inheritParams binary_elasticnet
#' @inheritParams binary_lgbm
#' @param elastic_workers integer number of cores to use for parallelized tuning of elasticnet
#' @param lgbm_workers integer number of cores to use for parallelized tuning of trees
#'
#' @return A lostic elasticnet + gradient boosting model for binary outcomes,
#' with class 'learner'.
binary_elasticgbm = structure(
  class = c("learner_constructor", "function"),
  function(alpha            = seq(0, 1, length.out = 5),
           learning_rate    = 0.2,
           num_leaves       = 7,
           max_depth        = -1,
           min_data_in_leaf = 20,
           force_col_wise   = FALSE,
           metric           = 'rmse',
           nrounds          = 10000L,
           elastic_workers  = 1L,
           lgbm_workers     = 1L) {
    make_learner(name = "binary_elasticgbm",
                 tune_fun = purrr::partial(binary_elasticgbm_tune,
                                           alpha = alpha,
                                           learning_rate = learning_rate,
                                           num_leaves = num_leaves, 
                                           max_depth = max_depth,
                                           min_data_in_leaf = min_data_in_leaf,
                                           force_col_wise = force_col_wise,
                                           metric = metric, 
                                           nrounds = nrounds, 
                                           elastic_workers = elastic_workers,
                                           lgbm_workers = lgbm_workers),
                 predict_fun = binary_elasticgbm_predict,
                 predict_tuned_fun = NULL)
  }
)

binary_elasticgbm_tune = function(features, tgt, wt = rep(1, nrow(features)),
                                  tune_folds, 
                                  alpha,
                                  learning_rate, num_leaves, max_depth,
                                  min_data_in_leaf, force_col_wise,
                                  metric, nrounds, 
                                  elastic_workers,
                                  lgbm_workers) {
  
  tune_folds = match(tune_folds, sort(unique(tune_folds)))
  
  # Tune elasticnet ------------------------------------------------------------------------------
  if (elastic_workers > 1) {
    elastic_parallel = TRUE
    doFuture::registerDoFuture()
    future::plan(future::multicore, workers = elastic_workers)
  } else {
    elastic_parallel = FALSE
  }
  
  max_vars = min(min(table(tune_folds)) - 1,
                 ncol(features))
  
  elnet_tuning = purrr::map(alpha,
                            ~glmnet::cv.glmnet(x = features,
                                               y = tgt,
                                               weights = wt,
                                               foldid  = tune_folds,
                                               family  = "binomial",
                                               alpha   = .x,
                                               keep    = TRUE,
                                               parallel = elastic_parallel,
                                               pmax     = max_vars))
  
  cv_elnet_cvm = purrr::map_dbl(elnet_tuning, function(f) { min(f$cvm) })
  
  best_alpha   = alpha[which.min(cv_elnet_cvm)]
  best_elnet   = elnet_tuning[[ which.min(cv_elnet_cvm) ]]
  lgt_elnet    = best_elnet$fit.preval[ , match(best_elnet$lambda, best_elnet$lambda.min) ]
  lgt_elnet    = as.numeric(lgt_elnet)
  
  
  # Tune LGBM from elastic net ------------------------------------------------------------------
  lgbm_design = expand.grid(num_leaves = num_leaves,
                            max_depth  = max_depth,
                            min_data_in_leaf = min_data_in_leaf,
                            learning_rate    = learning_rate)
  
  # cap the number of leaves (2^lgbm_design$max_depth - 1)
  lgbm_design$num_leaves = ifelse(lgbm_design$max_depth <= 0,
                                  lgbm_design$num_leaves,
                                  pmin(lgbm_design$num_leaves,
                                       2^lgbm_design$max_depth - 1))
  
  # remove redundant combinations
  lgbm_design = unique(lgbm_design)
  
  lgbm_tuning = purrr::pmap(lgbm_design,
                            function(...) {
                              params = list(...)
                              params$num_threads    = lgbm_workers
                              params$objective      = 'binary'
                              params$force_col_wise = force_col_wise
                              params$metric         = metric
                              
                              quiet_cv = purrr::quietly(lightgbm::lgb.cv)
                              
                              lgb_data = lightgbm::lgb.Dataset(data       = features,
                                                               label      = tgt,
                                                               weight     = wt,
                                                               init_score = lgt_elnet)
                              
                              out = quiet_cv(params = params,
                                             data   = lgb_data,
                                             folds  = split(seq_along(tune_folds), tune_folds),
                                             nrounds = nrounds,
                                             early_stopping_rounds = 20L,
                                             verbose = -1)
                              
                              out$result
                            })
  
  lgbm_tuning_df = purrr::map(lgbm_tuning, tidy_lgbm_cv)
  lgbm_tuning_df = purrr::list_rbind(lgbm_tuning_df,
                                     names_to = "idx_design")
  
  lgbm_best_param_row = which.min(lgbm_tuning_df$metric)
  lgbm_best_fit_iter  = lgbm_tuning_df$idx_design[lgbm_best_param_row]
  lgbm_best_fit = lgbm_tuning[[lgbm_best_fit_iter]] 
  
  
  # Return ------------------------------------------------------------------
  tune_res = list(
    train_fun   = purrr::partial(binary_elasticgbm_train,
                                 elnet_lambda = best_elnet$lambda,
                                 elnet_s      = best_elnet$lambda.min,
                                 elnet_alpha  = best_alpha,
                                 elnet_nfolds = length(unique(tune_folds)),
                                 lgbm_params  = lgbm_tuning_df$params[[lgbm_best_param_row]],
                                 lgbm_nrounds = lgbm_tuning_df$nrounds[[lgbm_best_param_row]])
  )
  
  tune_res
}

binary_elasticgbm_train = function(features, tgt, wt,
                                   elnet_lambda, elnet_s, elnet_alpha, elnet_nfolds,
                                   lgbm_params, lgbm_nrounds) {
  
  elnet_fit = glmnet::glmnet(x = features,
                             y = tgt,
                             weights = wt,
                             family  = "binomial",
                             alpha   = elnet_alpha,
                             lambda  = elnet_lambda)
  
  cv_elnet_fit = glmnet::cv.glmnet(x = features,
                                   y = tgt,
                                   weights = wt,
                                   nfolds  = elnet_nfolds,
                                   family  = "binomial",
                                   keep    = TRUE,
                                   alpha   = elnet_alpha,
                                   lambda  = elnet_lambda)
  
  lgt_elnet_oof = cv_elnet_fit$fit.preval[ , match(elnet_s, cv_elnet_fit$lambda) ]
  
  lgbm_fit = lightgbm::lightgbm(data = lightgbm::lgb.Dataset(data  = features, 
                                                             label = tgt, 
                                                             weight = wt,
                                                             init_score = lgt_elnet_oof),
                                params     = lgbm_params,
                                nrounds    = lgbm_nrounds,
                                verbose    = -1)
  
  list(elnet = elnet_fit,
       lgbm  = lgbm_fit,
       elnet_alpha = elnet_alpha,
       elnet_s     = elnet_s,
       lgbm_params = c(lgbm_params, nrounds = lgbm_nrounds))
  
}

binary_elasticgbm_predict = function(model, features) {
  
  lgt_elnet = stats::predict(model$elnet, features, s = model$elnet_s, type = "link")
  lgt_lgbm  = stats::predict(model$lgbm, features, type = "raw")
  
  as.numeric(stats::plogis(lgt_elnet + lgt_lgbm))
}


tidy_lgbm_cv = function(result) {
  
  params = result$boosters[[1]]$booster$params
  params$metric = params$metric[[1]]
  params$early_stopping_round = NULL
  
  data.frame(params  = I(list(params)),
             nrounds = result$best_iter,
             metric  = result$best_score)
}