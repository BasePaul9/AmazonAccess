####                 AMAZON EMPLOYEE ACCESS COMPETITION                 ####

library(tidyverse)
library(tidymodels)
install.packages("embed")
library(embed)
install.packages("vroom")
library(vroom)
install.packages("skimr")
library(skimr)

# setwd("./AmazonAccess")
test <- vroom("./test.csv")
train <- vroom("./train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

# skim(train)

# ggplot(data = train, mapping = aes(x = MGR_ID, y = ROLE_TITLE)) +
#   geom_point()
# 
# ggplot(data = train) +
#   geom_boxplot(aes(y = MGR_ID))
# ggplot(data = train) + 
#   geom_boxplot(aes(y = RESOURCE))


my_recipe <- recipe(ACTION ~., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())

# apply the recipe to your data
prep <- prep(my_recipe)
baked_data <- bake(prep, new_data = train)

####                 PENALIZED LOGISTIC REGRESSION                 ####

my_recipe <- recipe(ACTION ~., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

pen_logit_model <- logistic_reg(mixture = tune(),
                                penalty = tune()) %>%
  set_engine("glmnet")

pen_log_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pen_logit_model)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats=1)

cv_results <- pen_log_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- cv_results %>%
  select_best()

final_wf <- pen_log_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

pen_log_preds <- final_wf %>%
  predict(new_data = test, type = "prob")

kag_sub <- pen_log_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) %>% #rename pred to count (for submission to Kaggle)
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./PenalizedLogitModel.csv", delim=",")