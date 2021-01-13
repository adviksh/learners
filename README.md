# learners: tuning, training, and prediction for supervised learning models

This package implements various machine learning algorithms in a standardized format. It exists mainly as a helper package for [moxie](https://github.com/adviksh/moxie), and is roughly inspired by Python's scikit-learn framework.

If you're looking to run your own ML workflows, I'd recommend an actively maintained framework like [mlr3](https://mlr3.mlr-org.com) or [tidymodels](https://www.tidymodels.org). I wrote `learners` because both of these frameworks were still maturing in early 2018, when I started work on `moxie`. If I had to start from scratch in 2021, I'd probably use one of them.

## Install

### Option One: R
Make sure you have an up-to-date version of the `devtools` package. Then, open an R session and run the following command:
```
devtools::install_github("adviksh/learners")
```

### Option Two: Command line
From a terminal window, navigate to the location where you want to download this repo to. Then run the following commands:
```
git clone git@github.com:adviks/learners
Rscript -e "devtools::install(pkg = 'learners', build_vignettes = TRUE)"
```

## Usage
The script below gives a bare-bones introduction to model creation, tuning, training, and prediction.

```r
library(learners)

# a "learner" is a predictive model
# this package includes functions to construct some common learners
# the `binary_...()` family of functions construct binary classifiers
# ex: binary_elasticnet is a function that creates a logistic regression model
# with an elasticnet penalty
class(binary_elasticnet)

# calling binary_elasticnet() will return a "learner"
elastic_learner <- binary_elasticnet()
class(elastic_learner)

# we'll construct some simple data with the `mtcars` dataset
# the features have to be a matrix
x <- as.matrix(mtcars[,c("mpg", "cyl")])

# for binary problems, the target has to be 0/1 or factor with two levels
# if a factor, the second level is assumed to be the positive class
y <- mtcars[,c("vs")]

# we also hae to create folds to use for cross-validation when tuning
tune_folds <- rep_len(1:8, nrow(x))

# the tune() function sets hyperparameters for a learner,
# based on the provided data
tuned_elastic_learner <- tune(elastic_learner,
                              features = x,
                              tgt = y,
                              tune_folds = tune_folds)

# once a learner has been tuned, the train() function fits the model to the
# provided data
trained_elastic_learner <- train(tuned_elastic_learner,
                                 features = x,
                                 tgt = y)

# once a learner has been trained, the predict() function returns predicted
# values for the provided data. for binary classification, the predictions are
# always the predicted probability of being in the positive class
predictions <- predict(trained_elastic_learner, x)

```
