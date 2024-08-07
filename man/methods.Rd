% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/methods.R
\name{tune}
\alias{tune}
\alias{train}
\alias{tune_predict_ins}
\alias{predict_oos}
\alias{tune_predict_oos}
\alias{tune_predict_oos_fold}
\alias{train_predict_oos_fold}
\title{Methods}
\usage{
tune(object, features, tgt, wt = NULL, tune_folds, ...)

train(object, features, tgt, wt = NULL, ...)

tune_predict_ins(object, features, tgt, wt = NULL, tune_folds, ...)

predict_oos(object, features, tgt, wt = NULL, tune_folds, ...)

tune_predict_oos(object, features, tgt, wt = NULL, tune_folds, ...)

tune_predict_oos_fold(
  object,
  features,
  tgt,
  wt = NULL,
  tune_folds,
  which_fold,
  ...
)

train_predict_oos_fold(
  object,
  features,
  tgt,
  wt = NULL,
  tune_folds,
  which_fold,
  ...
)
}
\arguments{
\item{object}{a model}

\item{features}{input data, where each observation is a row}

\item{tgt}{the variable to predict}

\item{wt}{optional, weights for each observation}

\item{tune_folds}{integer vector, a division of data into cross-validation folds}

\item{...}{additional arguments}

\item{which_fold}{for tune_predict_oos_fold, which fold to retrun out of sample
predictions for.}
}
\description{
Methods
}
