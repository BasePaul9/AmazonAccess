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


####                 LOGISTIC REGRESSION                 ####

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

vroom_write(x=kag_sub, file="./LogitModel.csv", delim=",")

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


####                 BATCH & PARALLEL COMPUTING                 ####

library(doParallel)

parallel::detectCores()

cl <- makePSOCKcluster(4)

registerDoParallel(cl)

#CodeGoesHere

stopCluster(cl)


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

####                 CLASSIFICATION RANDOM FORESTS                 ####


rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range = (c(1,(ncol(baked_data))-1))),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train, v = 5, repeats=1)

cv_results <- rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- cv_results %>%
  select_best()

final_wf <- rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

rf_preds <- predict(final_wf, new_data = test, type="prob")

kag_sub <- rf_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) %>% #rename pred to count (for submission to Kaggle)
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./RandForestModel.csv", delim=",")


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


####                 PRINCIPAL COMPONENTS ANALYSIS                  ####

my_recipe <- recipe(ACTION ~., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.8) #Threshold is between 0 and 1


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


####                 BALANCING DATA                  ####

my_recipe <- recipe(ACTION ~., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 150)

baked_data <- bake(prep(my_recipe), new_data = train)














