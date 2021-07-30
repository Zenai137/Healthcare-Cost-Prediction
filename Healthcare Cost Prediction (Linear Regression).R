#------------------------------------------------------------------------------#
#   Health Care Cost Prediction
#   Linear Regression
#------------------------------------------------------------------------------#

#set Working Directory
getwd()
setwd("C:/Users/medma/Documents/Data Science/rcode")

#load needed libraries
library(ggplot2)
library(dplyr)
library(Hmisc)
library(cowplot)
library(WVPlots)
set.seed(123)

#load the data
ins <- read.csv('../data/insurance.csv')

#inspect the data
sample_n(ins, 5)

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#
#Data Dictionary
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#

#Age: insurance contractor age, years
#Sex: insurance contractor gender, [female, male]
#BMI: Body mass index, providing an understanding of body, weights that are 
#  relatively high or low relative to height, objective index of body weight 
#  (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
#Children: number of children covered by health insurance / Number of dependents
#Smoker: smoking, [yes, no]
#Region: the beneficiary's residential area in the US,
#  [northeast, southeast, southwest, northwest]
#Charges: Individual medical costs billed by health insurance, 
#  $ #predicted value

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#
#Exploratory Data Analysis
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#

#inspect descriptive statistics of the data
describe(ins)

#bivariate plot
#correlations between age/bmi and charges
x <- ggplot(ins, aes(age, charges)) +
  geom_jitter(color = "#00AEDB", alpha = 0.5) +
  theme_light()

y <- ggplot(ins, aes(bmi, charges)) +
  geom_jitter(color = "#00B159", alpha = 0.5) +
  theme_light()

p <- plot_grid(x, y) 
title <- ggdraw() + draw_label("1. Correlation between Charges and Age / BMI", 
                               fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))
#age and charges have a positive relationship, while bmi and charges have a 
#slightly positive relationship

#correlations between sex/children covered by insurance
x <- ggplot(ins, aes(sex, charges)) +
  geom_jitter(aes(color = sex), alpha = 0.7) +
  theme_light()

y <- ggplot(ins, aes(children, charges)) +
  geom_jitter(aes(color = children), alpha = 0.7) +
  theme_light()

p <- plot_grid(x, y) 
title <- ggdraw() + draw_label(
  "2. Correlation between Charges and Sex / Children covered by insurance", 
                               fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))
#no obvious relationship can be observed between sex and charges, but an
#interesting observation with children covered by insurance is that charges
#decrease for 4-5 children covered by insurance

#correlations between smoker/region and charges
x <- ggplot(ins, aes(smoker, charges)) +
  geom_jitter(aes(color = smoker), alpha = 0.7) +
  theme_light()

y <- ggplot(ins, aes(region, charges)) +
  geom_jitter(aes(color = region), alpha = 0.7) +
  theme_light()

p <- plot_grid(x, y) 
title <- ggdraw() + draw_label(
  "3. Correlation between Charges and Smoker / Region", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))
#charges are higher for smokers, and there is no obvious relationship between
#region and charges

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#
#Data Modeling: Linear Regression Model
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#

#use 80% of the data as the training set
n_train <- round(0.8 *nrow(ins))
train_indicies <- sample(1:nrow(ins), n_train)
ins_train <- ins[train_indicies, ]
ins_test <- ins[-train_indicies, ]

#variables for first model
formula_0 <- as.formula("charges ~ age + sex + bmi + 
                        children + smoker + region")

#train and test the first model
model_0 <- lm(formula_0, data=ins_train)
summary(model_0)
#only age, bmi, children, and smoker are have a significant effect on charges,
#so a new training model will be created with only those explanatory variables

#saving r-squared
r_sq_0 <- summary(model_0)$r.squared
#the r-squared shows that the model is highly accurate

#predict data on test set
prediction_0 <- predict(model_0, newdata = ins_test)
#calculating the residuals
residuals_0 <- ins_test$charges - prediction_0
#calculating Root Mean Squared Error
rmse_0 <- sqrt(mean(residuals_0^2))

#train and test a new model
formula_1 <- as.formula("charges ~ age + bmi + children + smoker")

model_1 <- lm(formula_1, data = ins_train)
summary(model_1)

#save the new r-squared
r_sq_1 <- summary(model_1)$r.squared

prediction_1 <- predict(model_1, newdata = ins_test)

residuals_1 <- ins_test$charges - prediction_1
rmse_1 <- sqrt(mean(residuals_1^2))

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#
#Model Performance Evaluations
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#

#compare the r-squared values
print(paste0("R-squared for first model: ", round(r_sq_0, 4)))
print(paste0("R-squared for new model: ", round(r_sq_1, 4)))

#compare the root mean squared error values
print(paste0("RMSE for first model: ", round(rmse_0, 2)))
print(paste0("RMSE for new model: ", round(rmse_1, 2)))

#the models have highly similar performance metric values, 
#so either one can be used.

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#
#Model Performance Evaluations Visualized
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#

#prediction values vs real values
ins_test$prediction <- predict(model_1, newdata = ins_test)
ggplot(data = ins_test, aes(x = prediction, y = charges)) + 
  geom_point(color = "#f37735", alpha = 0.7) + 
  geom_abline(color = "#d11141") +
  ggtitle("Prediction Values vs. Real values")

#prediction values vs residual values
ins_test$residuals <- ins_test$charges - ins_test$prediction

ggplot(data = ins_test, aes(x = prediction, y = residuals)) +
  geom_pointrange(aes(ymin = 0, ymax = residuals), color = "#f37735", alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = 3, color = "#d11141") +
  ggtitle("Prediction Values vs. Residual Values")

#histogram of residual values
ggplot(ins_test, aes(x = residuals)) + 
  geom_histogram(bins = 15, fill = "#edae49") +
  ggtitle("Histogram of Residuals")

#plot a gain curve
GainCurvePlot(ins_test, "prediction", "charges", "Model")

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#
#Application to New Data
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --#

#1. Bob: 19 years old, BMI 27.9, has no children, smokes, from northwest region.
#2. Lisa: 40 years old, BMI 50, 2 children, doesn't smoke, from southeast region.
#3. John: 30 years old. BMI 31.2, no children, doesn't smoke, from northeast region.

#Bob
Bob <- data.frame(age = 19,
                  bmi = 27.9,
                  children = 0,
                  smoker = "yes",
                  region = "northwest")
print(paste0("Health care charges for Bob: ", round(predict(model_1, Bob), 2)))

#Lisa
Lisa <- data.frame(age = 40,
                   bmi = 50,
                   children = 2,
                   smoker = "no",
                   region = "southeast")
print(paste0("Health care charges for Lisa: ", round(predict(model_1, Lisa), 2)))

#John
John <- data.frame(age = 30,
                   bmi = 31.2,
                   children = 0,
                   smoker = "no",
                   region = "northeast")
print(paste0("Health care charges for John: ", round(predict(model_1, John), 2)))

