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
  #mutate(count = exp(count)) %>%
  mutate(ACTION=pmax(0, ACTION)) 

vroom_write(x=kag_sub, file="./LogitModel.csv", delim=",")