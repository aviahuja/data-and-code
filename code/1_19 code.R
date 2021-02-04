# Introduction to Machine Learning (33002)
# Philip Waggoner (https://pdwaggoner.github.io/)

# Code for class

# kNN + ML Tasks

# load some libraries
library(tidyverse)
library(here)
library(patchwork)
library(tidymodels)

# load the 2016 ANES pilot study data
anes <- read_csv(here("data", "anes_pilot_2016.csv"))

# select some features and clean: party, and 2 fts
anes_short <- anes %>% 
  select(pid3, fttrump, ftobama) %>%      #keeps only certain variables
  mutate(democrat = as.factor(ifelse(pid3 == 1, 1, 0)),  #creates a dichotomous party affiliation variable/feature (if Dem, make it a 1, everything else make 0)
         fttrump = replace(fttrump, fttrump > 100, NA),  #get rid of NAs (any values > 100)
         ftobama = replace(ftobama, ftobama > 100, NA)) %>% #same thing for Obama
  drop_na()    #drop NA

anes_short <- anes_short %>% 
  select(-c(pid3)) %>%     # get rid of the original party ID variables
  relocate(c(democrat))    # move the new democrat to the front or "relocate"

anes_short %>% 
  glimpse()

# visualize the data to get a sense of the distribution (exploratory data analysis)
anes_short %>% 
  ggplot(aes(fttrump, ftobama, 
             col = democrat)) +   #coloring the points based on the dichotomous democrat feature
  geom_point() +
  theme_minimal()

# obama density
o_d <- anes_short %>% 
  ggplot(aes(ftobama, 
             col = democrat)) +
  geom_density() +       #density plot for feelings towards obama only
  ggtitle("Obama") +
  theme_minimal()

# trump density
t_d <- anes_short %>% 
  ggplot(aes(fttrump, 
             col = democrat)) +
  geom_density() +      #density plot for feelings towards trump only
  ggtitle("Trump") +
  theme_minimal()

# side by side both density plots
o_d + t_d

# Change shapes and colors for more descriptive plots if you want
anes_short %>% 
  ggplot(aes(fttrump, ftobama, 
             shape = democrat)) +   #shape plot instead of color plot
  geom_point() +
  theme_minimal()

anes_short %>% 
  ggplot(aes(fttrump, ftobama, 
             shape = democrat,     #both shape and color plot
             col = democrat)) +
  geom_point() +
  theme_minimal()

#IMPORTANT NOTE: kNN cannot handle categorical variables (features)

# scale continuous inputs (put everything on a common scale, to be safe)
anes_scaled <- anes_short %>% 
  mutate_at(c("fttrump", "ftobama"), #selects the variables we want to mutate (in this case, scale) alone
            ~(scale(.)))     #the dot is just shorthand for the data
# we are only scaling the input features not "democrat" the new feature we created
# if we wanted to scale everything we wouldn't do mutate_at, we would just say scale


## train the model on the full data (via tidymodels)
# define model type
mod <- nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

# fit (training part)
knn <- mod %>% 
  fit(democrat ~ .,         # read it as fitting a model with democrat on the LHS and dot ',' is just shorthand for all the features that exist 
      data = anes_scaled)   # this is the data object we are using for framing
# If we didn't want to include all the features we could specify as fit(democrat ~ fttrump + ftobama)

# eval
knn %>% 
  predict(anes_scaled) %>%   # predict based on the training data (predict is a base R function)
  bind_cols(anes_scaled) %>% # allows us to stack stuff up (same as cbind) - just an aesthetic goal to have the print results appear more nicely
  metrics(truth = democrat,  # these are the labels we want to pass to the prediction (telling it the ground truth about who the democrats are)
          estimate = .pred_class)  # telling it what the predicted classes are

# accuracy is just the accuracy rate (the % of correct classifications that we got - here 86.6%)
# kap is kappa statistic (a null hypothesis version of the accuracy) - can ignore


# predict and visualize the accuracy of our classification solution (kNN fit to the full data)
knn %>% 
  predict(anes_scaled) %>% 
  mutate(model = "knn", 
         truth = anes_scaled$democrat) %>% 
  mutate(correct = if_else(.pred_class == truth, "Yes", "No")) %>% 
  ggplot() +
  geom_bar(alpha = 0.8, aes(correct, fill = correct)) + 
  labs(x = "Correct?",
       y = "Count",
       fill = "Correctly\nClassified") +
  theme_minimal()


# k-fold CV
## first split
set.seed(1234)

split <- initial_split(anes_scaled,
                       prop = 0.70) #set 70% of the data aside for our training set, remaining 30% for testing
train <- training(split)   
test <- testing(split)

## kFold Cross Validation (R just calls the function vfold_cv) 
cv_train <- vfold_cv(train, # set up this cannister of a bunch of different datasets for 10-fold cross validation
               v = 10)

cv_train
# When we print, we can see that each one of the folds, 753 or 70% of the obs get assigned to the training set and 30% are set aside for the testing set

## Now, create a recipe to make things easier
recipe <- recipe(democrat ~ ., 
                 data = anes_scaled)  #same as feature engineering, refer to Lecture 2 for details

# define model type from earlier, but with `k` addition
mod_new <- nearest_neighbor() %>% 
  set_args(neighbors = tune()) %>%  #we are going to TUNE this hyperparameter neighbors (k) - this is currently empty
  set_engine("kknn") %>%     #kknn is just the package
  set_mode("classification") 

# Define a workflow; This is just a way to keep things neat and tidy (pun intended)
workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(mod_new)

# Now, we tune() instead of fit()
grid <- expand.grid(neighbors = c(1:25))   #define a grid using base R (creates a vector from 1 to 25)

res <- workflow %>%
  tune_grid(resamples = cv_train,    #for the resamples, we pass to it our cross-validation cannister
            grid = grid,
            metrics = metric_set(roc_auc, accuracy))  #we want to use the area under the curve (ROC) and are interested in the raw accuracy rate
 
# inspect 
res %>% 
  collect_metrics(summarize = TRUE) 

res %>% 
  select_best(metric = "roc_auc")   # just printing the best ROC iteration

res %>% 
  select_best(metric = "accuracy")   #hjust printing the highest accuracy iteration

# final/best
final <- res %>% 
  select_best(metric = "roc_auc")

# Updating the workflow by passing the final model on the basis of the best metric
workflow <- workflow %>%
  finalize_workflow(final)

# train and eval in one 
final_mod <- workflow %>%
  last_fit(split)   #here we take our original split to train and predict simultaneously using last_fit
# super powerful function

# inspect
final_mod %>% 
  collect_predictions() #show predictions made on test set
  
final_mod %>%  
  collect_metrics()  #show model-level metrics based on those we specified in tune_grid() ie. roc_aud and accuracy
# We can see that we did well - we correctly (accurately) predicted 77% of the party labels on the basis of these two feeling thermometers
# This is a little worse than we did earlier (when it was 86%), but we had previously trained and tested on the 
# same set of data. Now after the split the predictions are a bit worse, but that's the right way to do it.
# The other earlier estimate is overly optimistics because we trained and tested the same data.

# create confusion matrix based on predictions from the model (another way to present results)
final_mod %>% 
  collect_predictions() %>% 
  conf_mat(truth = democrat, 
           estimate = .pred_class,
           dnn = c("Pred", "Truth"))

# visualization, of course (another way of presenting the accuracy of your model's predictions)
final_mod %>% 
  collect_predictions() %>% 
  ggplot() +
  geom_bar(aes(x = .pred_class, 
                   fill = democrat)) +
  facet_wrap(~ democrat) +
  labs(x = "Predicted Party Affiliations", 
       fill = "Democrat",
       caption = "Note: facets are ground truth\nFor '1' in truth of '0' (53), kNN predicted incorrectly\nVice verse for the other class (29), via confusion matrix") +
  theme_minimal()
