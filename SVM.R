####                 AMAZON EMPLOYEE ACCESS COMPETITION                 ####

library(tidyverse)
library(tidymodels)
install.packages("embed")
library(embed)
install.packages("vroom")
library(vroom)
install.packages("skimr")
library(skimr)
install.packages("kernlab")
library(kernlab)

#setwd("./AmazonAccess")
test <- vroom("./test.csv")
train <- vroom("./train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

my_recipe <- recipe(ACTION ~., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors())

####                 SUPPORT VECTOR MACHINES                  ####

svmRadial <- svm_rbf(rbf_sigma = tune(), 
                     cost = tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train, v = 5, repeats=1)

cv_results <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- cv_results %>%
  select_best()

final_wf <- svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

svm_preds <- predict(final_wf, new_data = test, type="prob")

kag_sub <- svm_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) %>% #rename pred to count (for submission to Kaggle)
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./SupportVectorMach.csv", delim=",")


















