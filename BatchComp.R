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

####                 BATCH & PARALLEL COMPUTING                 ####

library(doParallel)

parallel::detectCores()

cl <- makePSOCKcluster(4)

registerDoParallel(cl)

#CodeGoesHere

stopCluster(cl)