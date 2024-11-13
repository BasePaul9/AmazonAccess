####                 AMAZON EMPLOYEE ACCESS COMPETITION                 ####

library(tidyverse)
library(tidymodels)
install.packages("embed")
library(embed)
install.packages("vroom")
library(vroom)
install.packages("themis")
library(themis)

#setwd("./AmazonAccess")
test <- vroom("./test.csv")
train <- vroom("./train.csv") %>%
  mutate(ACTION = as.factor(ACTION))


####                 BALANCING DATA                  ####

my_recipe <- recipe(ACTION ~., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.8) %>%
  step_smote(all_outcomes(), neighbors = 150)

baked_data <- bake(prep(my_recipe), new_data = train)

#### Logistic Reg ####

logit_model <- logistic_reg() %>%
  set_engine("glm")

logit_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logit_model) %>%
  fit(data = train)

logit_preds <- predict(logit_wf,
                       new_data = test,
                       type = "prob")

kag_sub <- logit_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) %>% #rename pred to count (for submission to Kaggle)
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./BalancedLogitModel.csv", delim=",") # score = 0.67471

#### Penalized Logit ####

pen_logit_model <- logistic_reg(mixture = 0.25,
                                penalty = 0.00316) %>%
  set_engine("glmnet")

pen_log_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pen_logit_model) %>%
  fit(data = train)

pen_log_preds <- pen_log_wf %>%
  predict(new_data = test, type = "prob")

kag_sub <- pen_log_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) %>% #rename pred to count (for submission to Kaggle)
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./BalancedPenalizedModel.csv", delim=",") # score = 0.67560

#### KNN ####

knn_model <- nearest_neighbor(neighbors = round(sqrt(nrow(train)))) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model) %>%
  fit(data = train)

knn_preds <- predict(knn_wf, new_data = test, type="prob")

kag_sub <- knn_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) %>% #rename pred to count (for submission to Kaggle)
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./BalancedKNNModel.csv", delim=",") # score = 0.74578

#### Class RF ####

rf_mod <- rand_forest(mtry = 9,
                      min_n = 20,
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod) %>%
  fit(data = train)

rf_preds <- predict(rf_wf, new_data = test, type="prob")

kag_sub <- rf_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) %>% #rename pred to count (for submission to Kaggle)
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./BalancedRandForestModel.csv", delim=",") # score = 0.76974

#### Naive Bayes ####

install.packages("discrim")
library(discrim)

nb_model <- naive_Bayes(Laplace = 0,
                        smoothness = 0.5) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model) %>%
  fit(data = train)

nb_preds <- predict(nb_wf, new_data = test, type="prob")

kag_sub <- nb_preds %>%
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION=.pred_1) %>% 
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./BalancedNaiveBayesModel.csv", delim=",") # score = 0.65188




