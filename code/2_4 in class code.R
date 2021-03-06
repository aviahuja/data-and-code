---
  title: "Non-Linear Regression"
author: "Philip Waggoner, MACS 30100 <br /> University of Chicago"
output: pdf_document
---
  
  # Overview
  
  Let's fit some non-linear models, but based on some of the examples and data from the book.

The methods: 

  - Polynomial Regression
  - Generalized Additive Models (GAMs)

# Polynomial Regression

Let's start by fitting a simple polynomial regression to the `Wage` data.

```{r}
# load some core packages
library(ISLR) # for some toy data
library(tidyverse) # for data cleaning and plotting
library(tidymodels) # for splitting
library(broom) # for model summaries
library(splines) # for ns()
library(gam) # for GAM
# fit a simply polynomial model
poly_mod <- lm(wage ~ poly(age, 4, raw = TRUE), 
               data = Wage)
tidy(poly_mod)
```

Predict the function over the range of age.

```{r}
# Get min/max values of age
agelims <- Wage %>%
  select(age) %>%
  range()
# Generate a sequence of age values spanning the range
age_grid <- seq(from = min(agelims), 
                to = max(agelims))
# Predict across ages along with the SEs (useful for plotting next)
preds <- predict(poly_mod, 
                 newdata = list(age = age_grid), 
                 se = TRUE)
```

Plot.

```{r}
ggplot() +
  geom_point(data = Wage, aes(x = age, y = wage)) +
  geom_line(aes(x = age_grid, y = preds$fit), color = "red", size = 1.5, alpha = 0.7) +
  geom_ribbon(aes(x = age_grid, 
                  ymin = preds$fit - 2 * preds$se.fit, 
                  ymax = preds$fit + 2 * preds$se.fit), 
              alpha = 0.3) +
  xlim(agelims) +
  labs(title = "4th Order Polynomial Regression over the Range of Age") + 
  theme_minimal()
```

## Picking $d$

How should we pick the polynomial order? Maybe for the model that minimizes test error.

For this, we will use another ISLR data set, `Auto`.

```{r}
# load the data
Auto <- as_tibble(Auto)
# a simple model (fit to the full data) shows definite non-linearity (violation of normally distributed error variance; so a SLM won't do a good job)
ggplot(Auto, aes(horsepower, mpg)) +
  geom_point(alpha = .1) + 
  geom_smooth(method = "lm", se = TRUE) + 
  theme_minimal()
```

We can do better... let's start with splitting

```{r}
set.seed(1234)
auto_split <- initial_split(data = Auto, 
                            prop = 0.8)
auto_train <- training(auto_split)
auto_test <- testing(auto_split)
```

Now, the range of $d$ to check.

```{r}
ggplot(Auto, aes(horsepower, mpg)) +
  geom_point(alpha = .1) +
  geom_smooth(aes(color = "1"),
              method = "glm",
              formula = y ~ poly(x, i = 1, raw = TRUE),
              se = FALSE) +
  geom_smooth(aes(color = "2"),
              method = "glm",
              formula = y ~ poly(x, i = 2, raw = TRUE),
              se = FALSE) +
  geom_smooth(aes(color = "3"),
              method = "glm",
              formula = y ~ poly(x, i = 3, raw = TRUE),
              se = FALSE) +
  geom_smooth(aes(color = "4"),
              method = "glm",
              formula = y ~ poly(x, i = 4, raw = TRUE),
              se = FALSE) +
  geom_smooth(aes(color = "5"),
              method = "glm",
              formula = y ~ poly(x, i = 5, raw = TRUE),
              se = FALSE) +
  scale_color_brewer(type = "qual", palette = "Dark2") +
  labs(x = "Horsepower",
       y = "MPG",
       color = "Polynomial\norder") +
  theme_minimal()
```

Great, but which is *best*? 

```{r}
# function to train the model (training set) and evaluate the model (testing set), across each polynomial regression fit
poly_results <- function(train, test, i) {
  mod <- glm(mpg ~ poly(horsepower, i, raw = TRUE), data = train)
  res <- augment(mod, 
                 newdata = test) %>%
    mse(truth = mpg, 
        estimate = .fitted)
  res
}
# function to return MSE for a unique order polynomial term
library(magrittr)
library(rcfss)
poly_mse <- function(i, train, test){
  poly_results(train, test, i) %$%
    mean(.estimate)
}
cv_mse <- tibble(terms = seq(from = 1, to = 5),
                 mse_test = map_dbl(terms, poly_mse, auto_train, auto_test))
ggplot(cv_mse, aes(terms, mse_test)) +
  geom_line() +
  labs(title = "Evaluating quadratic linear models",
       subtitle = "Using validation set",
       x = "Highest-order polynomial",
       y = "Mean Squared Error") +
  theme_minimal()
```

So which is best? 

# Generalized Additive Models (GAMs)

Let's go back to the `Wage` data for this part.

We now fit a really simple GAM to predict wage using natural splines for `year` and `age`, and nothing `education` for education. 

```{r}
gam_mod1 <- lm(wage ~ ns(year, 4) + ns(age, 5) + education, 
               data = Wage)
```

Let's try smoothing splines with `gam()` from `gam`.

```{r}
library(gam)
gam_mod2 <- gam(wage ~ s(year, 4) + s(age, 5) + education, 
                data = Wage)
par(mfrow = c(1,3))
plot(gam_mod2, 
     se = TRUE, 
     col = "red")
```

Summarize with `summary()`.

```{r}
summary(gam_mod2)
```

The $p$-values correspond to a null hypothesis of a linear relationship versus the alternative of a non-linear relationship. 

And of course, predict if we'd like, e.g.,

```{r}
preds <- predict(gam_mod2, 
                 newdata = Wage)
```