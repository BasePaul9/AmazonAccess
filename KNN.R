####                 AMAZON EMPLOYEE ACCESS COMPETITION                 ####

library(tidyverse)
library(tidymodels)
install.packages("embed")
library(embed)
install.packages("vroom")
library(vroom)


setwd("./AmazonAccess")
test <- vroom("./test.csv")
train <- vroom("./train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

my_recipe <- recipe(ACTION ~., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

# apply the recipe to your data
prep <- prep(my_recipe)
baked_data <- bake(prep, new_data = train)

####                 K-NEAREST NEIGHBORS                 ####

knn_model <- nearest_neighbor(neighbors = round(sqrt(nrow(train)))) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model) %>%
  fit(data = train)

# tuning_grid <- grid_regular(neighbors(),
#                             levels = 5)
# 
# folds <- vfold_cv(train, v = 5, repeats=1)
# 
# cv_results <- knn_wf %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# bestTune <- cv_results %>%
#   select_best()
# 
# final_wf <- knn_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = train)

knn_preds <- predict(knn_wf, new_data = test, type="prob")

kag_sub <- knn_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) %>% #rename pred to count (for submission to Kaggle)
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./KNNModel.csv", delim=",")

