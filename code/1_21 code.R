# Introduction to Machine Learning (33002)
# Philip Waggoner (https://pdwaggoner.github.io/)

# Code for class (logistic regression and LDA; classification pt 2)

# load some libraries
library(tidyverse)
library(here)
library(patchwork)
library(tidymodels)

# load the 2016 ANES pilot study data
anes <- read_csv(here("data", "anes_pilot_2016.csv"))

# select some features and clean: party, and 2 fts
anes_short <- anes %>% 
  select(pid3, fttrump, ftobama) %>% 
  mutate(democrat = as.factor(ifelse(pid3 == 1, 1, 0)),    #this is just simplifying the party ID to create just two levels through a new dummy for Democrats
         fttrump = replace(fttrump, fttrump > 100, NA),    #removing extreme/nonsensical values
         ftobama = replace(ftobama, ftobama > 100, NA)) %>% #same
  drop_na()

anes_short <- anes_short %>% 
  select(-c(pid3)) %>% 
  relocate(c(democrat))

anes_short %>% 
  skimr::skim()    #mostly to check the missing data and completion rate


# model fitting via tidymodels
# define mod and engine
mod <- logistic_reg() %>% # this is the only thing that changes from last time (instead of using k-nearest neighbors from last time)
  set_engine("glm") %>%   # glm means generalized linear models (logit, tobit, negative binomial, poisson etc.)
  set_mode("classification")

# fit 
logit <- mod %>%    #creating a canniester for what our model will do 
  fit(democrat ~ ., # predict whether democrat based on all features
      data = anes_short)
logit 

# eval
logit %>% 
  predict(anes_short) %>% 
  bind_cols(anes_short) %>% 
  metrics(truth = democrat,
          estimate = .pred_class)
# accuracy is about 80%

# predict and viz (same bar plot as last time)
logit %>% 
  predict(anes_short) %>% 
  mutate(model = "logit", 
         truth = anes_short$democrat) %>% 
  mutate(correct = if_else(.pred_class == truth, "Yes", "No")) %>% 
  ggplot() +
  geom_bar(alpha = 0.8, aes(correct, fill = correct)) + 
  labs(x = "Correct?",
       y = "Count",
       fill = "Correctly\nClassified") +
  theme_minimal()

# explore some of the output
library(broom)

tidy(logit)   # converts the regression output into neat table
# logged odds are more difficult to naturally interpret 
# odds ratio of 1.3 means there is a 30% greater chance of being a democrat (1) relative to a non-democrat (0)
# odds ratio of 0.8 means there is a 20% greater chance of being a nondemocrat (0) relative to a democrat (1)
# odds unlike probabilities can extend beyond 1. 
# Odds ratio of 3.4 means there is a 240% greater chance of being a 1 than a 0.

# Predicted probabilities
dont_like_trump <- tibble(fttrump = 0:10,
                          ftobama = mean(anes_short$ftobama))  #constructing a probablity range for people who really don't like trump

predicted_probs <- predict(logit,     #based on the predict function (estimate = .predict_class we defined above)
                           dont_like_trump, 
                           type = "prob")  #give us a probability here (we could change it to "class" instead aldo)
# visualize results
dont_like_trump %>%
  bind_cols(predicted_probs) %>%
  ggplot(aes(x = fttrump, 
             y = .pred_1)) +
  geom_point() +
  geom_errorbar(aes(ymin = (.pred_1) - sd(.pred_1), 
                    ymax = (.pred_1) + sd(.pred_1)), 
                width = 0.2) +
  geom_hline(yintercept = 0.50, linetype = "dashed") +
  ylim(0, 1) +
  labs(x = "Feelings toward Trump",
       y = "Probability of Being a Democrat") + 
  theme_minimal()
 
# hmm... what happened? Slightly lower probability of democrat as feelings towards trump increase
# notice that every one of these points is below the 50% cut point (ie. everyone is still a nondemocrat)
# in 2016, when the ANES was fielded, obama wasn't very popular. Just because someone has average feelings towards obama
# but hates trump doesn't necessarily mean they belong to the democrat class.

dont_like_trump_love_obama <- tibble(fttrump = 0:10,
                                     ftobama = 90:100)  # creating a new synthetic dataset which is now looking at hates trump but loves obama sample

predicted_probs_new <- predict(logit, 
                               dont_like_trump_love_obama, 
                               type = "prob")  #pass our new synthetic dataset to our trained model to get back predicted probabilities of being a democrat based on feelings towards the two
# visualize results
dont_like_trump_love_obama %>%
  bind_cols(predicted_probs_new) %>%
  ggplot(aes(x = fttrump, 
             y = .pred_1)) +
  geom_point() +
  geom_errorbar(aes(ymin = (.pred_1) - sd(.pred_1), 
                    ymax = (.pred_1) + sd(.pred_1)), 
                width = 0.2) +
  geom_hline(yintercept = 0.50, linetype = "dashed") +
  ylim(0, 1) +
  labs(x = "Feelings toward Trump",
       y = "Probability of Being a Democrat") + 
  theme_minimal()
# sure enough, now everyone is classified as a democrat (>0.5 probability of being democrat)

# However, we have not used cross validation  here.
# however, here we are training on the full set of data, but technically still passing new data
# we have learned patterns by feeding a synthetic (and specific) set of observations to the model (obama lovers, trump haters)
# Important thing is we are not training and testing on the same data (so we are not violating this principle)

# Cross validating a final, full model 
# Logit via kNN approach with tidymodels from last class
## split
set.seed(1234)

split <- initial_split(anes_short,
                       prop = 0.70) # 70% in the training set
train <- training(split) #storing them as separate sets of data
test <- testing(split)

cv_train <- vfold_cv(train, 
                     v = 10)  #define the resamples for the cross validation using just the training dataset
# the other 30% of the data (testing set) is being used to validate the data --> not being used for CV

## Now, create a recipe
recipe <- recipe(democrat ~ ., 
                 data = anes_short) 


# define mod and engine
mod <- logistic_reg() %>% # this is the only thing that changes from last time
  set_engine("glm") %>% 
  set_mode("classification")


# Define a workflow
workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(mod)

res <- workflow %>%
  fit_resamples(resamples = cv_train,    #last time we had tune_grid() here instead of fit_resamples
                metrics = metric_set(roc_auc, accuracy)) #storing these metrics like last time
# because here we don't have any hyperparameters, so there is no grid we have created (no range of hyperparameters we are searching over)

# finalize workflow and evaluate
final <- res %>% 
  select_best(metric = "accuracy")

workflow <- workflow %>%
  finalize_workflow(final)  #finalize_workflow is where the training and evaluation happens simultaneously on the final data

final_mod <- workflow %>%
  last_fit(split)    #we evaluate our model on that initial split (on the 30% held out testing data)


# inspect (if desired)
final_mod %>% 
  collect_predictions() 

final_mod %>%  
  collect_metrics() 

# accuracy is slightly lower than before because we are learning/training on only 70% of the data 
# whereas earlier we were training on the full set of data (so we don't have the opportunity to learn as much more)
# we are okay with a little bit higher bias (lower accuracy), because it's more generalizable

# create confusion matrix
final_mod %>% 
  collect_predictions() %>% 
  conf_mat(truth = democrat, 
           estimate = .pred_class,
           dnn = c("Pred", "Truth"))

# bar plot like last time
logit_plot <- final_mod %>% 
  collect_predictions() %>% 
  ggplot() +
  geom_bar(aes(x = .pred_class, 
               fill = democrat)) +
  facet_wrap(~ democrat) +
  labs(title = "From Logit Fit",
       x = "Predicted Party Affiliations", 
       fill = "Democrat") +
  theme_minimal()
logit_plot

# Finally, LDA

library(discrim)

mod <- discrim_linear() %>% # this is the only thing we are changing, again
  set_engine("MASS") %>% 
  set_mode("classification")


# Define a workflow
workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(mod)

res <- workflow %>%
  fit_resamples(resamples = cv_train,    # we don't have to redefine the CV_training, since it's just splitting the data, not making any predictions
                metrics = metric_set(roc_auc, accuracy))

# finalize workflow and evaluate
final <- res %>% 
  select_best(metric = "accuracy")

workflow <- workflow %>%
  finalize_workflow(final)

final_mod <- workflow %>%
  last_fit(split) 


# inspect (if desired)
final_mod %>% 
  collect_predictions() 

final_mod %>%  
  collect_metrics() 

#LDA just did a little bit better

# create confusion matrix
final_mod %>% 
  collect_predictions() %>% 
  conf_mat(truth = democrat, 
           estimate = .pred_class,
           dnn = c("Pred", "Truth"))

# bar plot like last time
lda_plot <- final_mod %>% 
  collect_predictions() %>% 
  ggplot() +
  geom_bar(aes(x = .pred_class, 
               fill = democrat)) +
  facet_wrap(~ democrat) +
  labs(title = "From LDA Fit",
       x = "Predicted Party Affiliations", 
       fill = "Democrat") +
  theme_minimal()


# plotting side by side
library(patchwork)

logit_plot + lda_plot

############### A non-Tidy version of LDA ####################3
# a quick tangent: a non-tidy approach for those less excited about the tidy approach (no judgement of course)
library(MASS)

set.seed(1234)

samples <- sample(nrow(anes_short), 
                      size = 0.8*nrow(anes_short))  

train <- anes_short[samples, ]
test <- anes_short[-samples, ]

lda <- lda(democrat ~ .,
           data = train)
lda

# some checks for accuracy
democrat <- test$democrat # set aside for ease

lda_pred <- predict(lda, test) 

# first few
data.frame(lda_pred)[1:5,]

# confusion matrix
table(lda_pred$class, democrat)

# accuracy rate
mean(lda_pred$class == democrat)

# density viz
true <- ggplot() +
  geom_density(aes(lda_pred$x, 
                   col = democrat),
               alpha = 0.5,
               linetype = "solid") +
  ylim(0.0, 0.9) +
  labs(title = "True Density") +
  theme_minimal() +
  theme(legend.position = "none")
true

both <- ggplot() +
  geom_density(aes(lda_pred$x, 
                   col = lda_pred$class),
               alpha = 0.5,
               linetype = "dashed") +
  geom_density(aes(lda_pred$x, 
                   col = democrat),
               alpha = 0.5,
               linetype = "solid") +
  ylim(0.0, 0.9) +
  labs(title = "True and Predicted Densities",
       caption = "Solid line = True density\nDashed line = Predicted density") +
  theme_minimal() +
  theme(legend.position = "none")
both

# side by side
true + both

#
