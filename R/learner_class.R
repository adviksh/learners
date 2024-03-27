#' @title Learner Class
#' @name learner
#'
#' @description
#' learners represents predictive models with the \code{learner} class.
#' A learner has four components: a tuning function, a training function,
#' a fitted model object, and a prediction function. \code{make_learner} is a
#' utility for creating new learners. You'll need a tuning function and a
#' prediction function to create a new learner. The training function and
#' fitted model usually depend on the dataset, so you don't need them to
#' create a learner.
#'
#' @details
#' The tune, train, and predict functions take only data as arguments. This
#' means they do not not accept parameters or hyperparameters as inputs.
#' Instead, all parameters and hyperparameters must be defined inside these
#' functions. Specifically:
#'
#' The tuning function takes (\code{features}, \code{tgt}, \code{wt}, and \code{folds})
#' as arguments, and returns a training function. These exact names must be
#' used for the arguments in this exact order. If hyperparameters need
#' to be set, they should be baked into the resulting training function. For
#' example, a tuning function for penalized regression could use
#' cross-validation over the provided \code{folds} to choose a penalty parameter,
#' and then return a training function that estimates a regression model
#' with the chosen penalty.
#'
#' The training function takes (\code{features}, \code{tgt}, \code{wt}) as arguments
#' and returns a fitted model object. These exact names must be
#' used for the arguments in this exact order. If any parameters need to be set,
#' they should be baked into the fitted model object. For example, a training
#' function for a regression model should estiamte all necessary coefficients
#' and store them in a fitted model so that \code{predict} can access them
#' later.
#'
#' The prediction function takes (\code{fitted_model}, \code{features}) as arguments
#' and returns predicted values. You can name the arguments as you'd like, but
#' they must appear in this exact order. The returned predictions should be
#' the probability of being in the positive class for binary classification,
#' continuous predictions for regression, and a matrix of class probabilities
#' (one column per class) for multiclass classification.
#'
#' @param name A name for the learner.
#' @param tune_fun A tuning function.
#' @param train_fun A training function.
#' @param tuned_model A tuned model.
#' @param trained_model A trained model.
#' @param predict_fun A prediction function.
#' @param object (learner) A learner object, as created by make_learner()
#' @param features (numeric matrix)
#' @param tgt (vector)
#' @param wt (nonnegative vector)
#' @param newdata (matrix) A dataset to form predictions form.
#' @param tune_folds (integer vector)
#' @param which_fold (integer)
#' @param ... further arguments passed to or from other methods
NULL

#' @rdname learner
#' @export
make_learner <- function(name = NULL,
                         tune_fun = NULL,
                         train_fun = NULL,
                         tuned_model = NULL,
                         trained_model = NULL,
                         predict_fun = stats::predict,
                         predict_tuned_fun = NULL) {

  structure(list(name = name,
                 tune_fun = tune_fun,
                 train_fun = train_fun,
                 tuned_model = tuned_model,
                 trained_model = trained_model,
                 predict_fun = predict_fun,
                 predict_tuned_fun = predict_tuned_fun),
            class = c(name, "learner"))

}
