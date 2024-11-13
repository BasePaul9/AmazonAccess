####                 AMAZON EMPLOYEE ACCESS COMPETITION                 ####

library(tidyverse)
library(tidymodels)
install.packages("embed")
library(embed)
install.packages("vroom")
library(vroom)
install.packages("themis")
library(themis)
install.packages("dbarts")
library(dbarts)

#setwd("./AmazonAccess")
test <- vroom("./test.csv")
train <- vroom("./train.csv") %>%
  mutate(ACTION = as.factor(ACTION))


####                 BALANCING DATA                  ####

bart_recipe <- recipe(ACTION ~., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.6) %>%
  step_smote(all_outcomes(), neighbors = 100)

# baked_data <- bake(prep(bart_recipe), new_data = train)

bart_model <- parsnip::bart(
  mode = "classification",
  engine = "dbarts",
  trees = 500,
  prior_terminal_node_coef = .95,
  prior_terminal_node_expo = 2,
  prior_outcome_range = 2
)

bart_wf <- workflow() %>%
  add_recipe(bart_recipe) %>%
  add_model(bart_model) %>%
  fit(data = train)

bart_preds <- predict(bart_wf, new_data = test, type="prob")

kag_sub <- bart_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) %>% #rename pred to count (for submission to Kaggle)
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./BART.csv", delim=",")

