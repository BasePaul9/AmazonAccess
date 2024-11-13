####                 AMAZON EMPLOYEE ACCESS COMPETITION                 ####

library(tidyverse)
library(tidymodels)
install.packages("embed")
library(embed)
install.packages("vroom")
library(vroom)
install.packages("discrim")
library(discrim)


#setwd("./AmazonAccess")
test <- vroom("./test.csv")
train <- vroom("./train.csv") %>% 
  mutate(ACTION = as.factor(ACTION))

my_recipe <- recipe(ACTION ~., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))



# baked_data <- bake(prep(my_recipe), new_data = train)


####                 NAIVE BAYES                 ####

nb_model <- naive_Bayes(Laplace = 0,
                        smoothness = 0.5) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model) %>%
  fit(data = train)

# tuning_grid <- grid_regular(Laplace(),
#                             smoothness(),
#                             levels = 5) ## L^2 total tuning possibilities
# 
# folds <- vfold_cv(train, v = 5, repeats=1)
# 
# cv_results <- nb_wf %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# bestTune <- cv_results %>%
#   select_best()
# 
# final_wf <- nb_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = train)

nb_preds <- predict(nb_wf, new_data = test, type="prob")

kag_sub <- nb_preds %>%
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION=.pred_1) %>% 
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./NaiveBayesModel.csv", delim=",")
