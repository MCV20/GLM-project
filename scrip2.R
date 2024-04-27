## Load libraries
library(naniar)
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)
library(tidymodels)
library(naivebayes)
library(gtsummary)
library(nnet)
library(coefplot)
library(ggpubr)
library(pROC)
library(MASS)
library(stargazer)
library(brant)
library(DescTools)
library(ordinalNet)
library(pomcheckr)
library(ROSE)
library(randomForest)
theme_set(theme_minimal())

## Load data
dat <- read.csv("ObesityDataSet_raw_and_data_sinthetic.csv")
dat <- dat[1:498,] #take original data before SMOTE

## Data Preprocessing (covariates)
dat$CALC[dat$CALC == "Always"] <- "Frequently"
dat$CAEC[dat$CAEC == "Always"] <- "Frequently"
dat$MTRANS[dat$MTRANS == "Motorbike"] <- "Motor_Vehicle"
dat$MTRANS[dat$MTRANS == "Automobile"] <- "Motor_Vehicle"
dat$MTRANS[dat$MTRANS == "Bike"] <- "Walking/Bike"
dat$MTRANS[dat$MTRANS == "Walking"] <- "Walking/Bike"

dat$NObeyesdad <- ifelse(dat$NObeyesdad %in% c("Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"), "Obese", dat$NObeyesdad)
dat$NObeyesdad <- ifelse(dat$NObeyesdad %in% c("Overweight_Level_I", "Overweight_Level_II"), "Overweight", dat$NObeyesdad)
dat$NObeyesdad[dat$NObeyesdad == "Insufficient_Weight"] <- "Underweight"

## Factor variables
dat <- dat %>%
  mutate_if(is.character, as.factor)

dat <- dat %>% mutate(
  FCVC = factor(FCVC),
  NCP = factor(NCP),
  CH2O = factor(CH2O),
  FAF = factor(FAF),
  TUE = factor(TUE)
)

## Remove weight variable and rename fam history
dat <- dat %>% dplyr::select(-Weight)
dat <- dat %>% rename(fam_hist = family_history_with_overweight)

## Variable Selection by Lasso = pvalue in this case
## Fit reduced model
mod_polr_reduced <- polr(NObeyesdad ~ Age + Height + NCP + CH2O + FAF +TUE, 
                         data = dat, Hess=TRUE,method = "logistic")
summary(mod_polr_reduced) 
stargazer(mod_polr_reduced, type="text", style="apsr", single.row = T)

confints <- confint(mod_polr_reduced)


tidy(mod_polr_reduced,conf.int = T,p.values = T)


## Interactions
mod1 <- polr(NObeyesdad ~ Age*Height +TUE + NCP+FAF + CH2O, 
                  data = dat, Hess=TRUE,method = "logistic")
stargazer(mod1, type="text", style="apsr", single.row = T)
mod2 <- polr(NObeyesdad ~ Age+TUE+ Height + NCP*FAF + CH2O, 
             data = dat, Hess=TRUE,method = "logistic")
stargazer(mod2, type="text", style="apsr", single.row = T)

mod3 <- polr(NObeyesdad ~ Age+TUE+ Height + NCP+ FAF*CH2O, 
             data = dat, Hess=TRUE,method = "logistic")
stargazer(mod3, type="text", style="apsr", single.row = T)

mod4 <- polr(NObeyesdad ~ Age+ Height + NCP*FAF*CH2O + TUE, 
             data = dat, Hess=TRUE,method = "logistic")
stargazer(mod4, type="text", style="apsr", single.row = T)

## Model Comparisons (the reduced model with no interactions is prefered)
anova(mod_polr_reduced,mod1)
anova(mod_polr_reduced,mod2)
anova(mod_polr_reduced,mod3)
anova(mod_polr_reduced,mod4) # Best model is the one with no interactions

AIC(mod_polr_reduced,mod1,mod2,mod3,mod4)
BIC(mod_polr_reduced,mod1,mod2,mod3,mod4)

## Predictions
predictions <- predict(mod_polr_reduced, newdata = dat)
confusionMatrix(predictions,dat$NObeyesdad) #Still very bad we try oversampling


## Create new dataset with Oversampling
set.seed(123)
dat2 <- upSample(dat %>% dplyr::select(-NObeyesdad),dat$NObeyesdad)
dat2 <- dat2 %>% rename(NObeyesdad = "Class")

## variable Selection with LASSO
## Set model Matrix
X <- model.matrix(NObeyesdad ~., 
                  data = dat2,
                  contrasts.arg = list(Gender = contrasts(dat$Gender,contrasts = FALSE),
                                       fam_hist = contrasts(dat$fam_hist,contrasts = FALSE),
                                       FAVC = contrasts(dat$FAVC,contrasts = FALSE), 
                                       FCVC = contrasts(dat$FCVC,contrasts = FALSE),
                                       NCP = contrasts(dat$NCP,contrasts = FALSE),
                                       CAEC = contrasts(dat$CAEC,contrasts = FALSE),
                                       SMOKE = contrasts(dat$SMOKE,contrasts = FALSE),
                                       CH2O = contrasts(dat$CH2O,contrasts = FALSE),
                                       SCC = contrasts(dat$SCC,contrasts = FALSE),
                                       FAF = contrasts(dat$FAF,contrasts = FALSE),
                                       TUE = contrasts(dat$TUE,contrasts = FALSE),
                                       CALC = contrasts(dat$CALC,contrasts = FALSE),
                                       MTRANS = contrasts(dat$MTRANS,contrasts = FALSE)))

X <- X[,-1]
Y <- dat2$NObeyesdad
Y <- as.numeric(dat2$NObeyesdad)
Y <- as.factor(Y)
lasso.mod2 <- ordinalNet(X, Y, 
                         family="cumulative",
                         link="logit",
                         alpha = 1)
which.min(lasso.mod2$aic)
lasso.mod2$coefs[12,] #only removes Gender

## Prop odds model With variables selected by LASSO (remove only Gender)
lasso.mod.up <- polr(NObeyesdad ~., 
               data = dat2 %>% dplyr::select(-Gender), Hess=TRUE,method = "logistic")
summary(lasso.mod.up)
stargazer(lasso.mod.up, type="text", style="apsr", single.row = T)


## Variable Selection by P-value
mod.up <- polr(NObeyesdad ~., 
               data = dat2, Hess=TRUE,method = "logistic")
stargazer(mod.up, type="text", style="apsr", single.row = T)

mod.up_reduced <-  polr(NObeyesdad ~ fam_hist + FAVC + NCP + CAEC + FAF + TUE, 
                        data = dat2, Hess=TRUE,method = "logistic")
summary(mod.up_reduced)

## Interactions
mod.up_int1 <- polr(NObeyesdad ~ fam_hist + FAVC*NCP + CAEC + FAF + TUE, 
                    data = dat2, Hess=TRUE,method = "logistic")
stargazer(mod.up_int1, type="text", style="apsr", single.row = T)

mod.up_int2 <- polr(NObeyesdad ~ fam_hist + FAVC+NCP*CAEC + FAF + TUE, 
                    data = dat2, Hess=TRUE,method = "logistic")
stargazer(mod.up_int2, type="text", style="apsr", single.row = T)

mod.up_int3 <- polr(NObeyesdad ~ fam_hist + FAVC+NCP*FAF+CAEC + TUE, 
                    data = dat2, Hess=TRUE,method = "logistic")
stargazer(mod.up_int3, type="text", style="apsr", single.row = T)

mod.up_int4 <- polr(NObeyesdad ~ fam_hist + FAVC+NCP+FAF*CAEC + TUE, 
                    data = dat2, Hess=TRUE,method = "logistic")
stargazer(mod.up_int4, type="text", style="apsr", single.row = T)

mod.up_int5 <- polr(NObeyesdad ~ fam_hist + FAF+NCP*FAVC*CAEC + TUE, 
                    data = dat2, Hess=TRUE,method = "logistic")
stargazer(mod.up_int5, type="text", style="apsr", single.row = T)

anova(mod.up_reduced,mod.up_int1)
anova(mod.up_reduced,mod.up_int2)
anova(mod.up_reduced,mod.up_int3)
anova(mod.up_reduced,mod.up_int4)
anova(mod.up_reduced,mod.up_int5)

AIC(mod.up_int1,mod.up_int2,mod.up_int3,mod.up_int4,mod.up_int5)
BIC(mod.up_int1,mod.up_int2,mod.up_int3,mod.up_int4,mod.up_int5)

anova(mod.up_int1,mod.up_int5)
anova(mod.up_int2,mod.up_int5)
anova(mod.up_int3,mod.up_int5)
anova(mod.up_int4,mod.up_int5) 

## confint(mod.up_int5)

## Function to Plot IC
plot_ic <- function(x, title = "95% CI") {
  h <- as.data.frame(summary(x)$coefficients)
  h <- h %>% mutate(up = Value + `Std. Error`*1.96,
                    down = Value - `Std. Error`*1.96)
  h$vars <- rownames(h)
  
  ggplot(h, aes(x = Value, y = vars)) +
    geom_point() +
    geom_segment(aes(x = up, xend = down, yend = vars)) +
    geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
    ggtitle(title)
}
plot_ic(mod.up_int5)

## Model Diagnostics on mod.up_int5


## (1) The Proportional Odds Assumption
##------------------------------------------------------------------------------
#test that the proportional odds assumption holds
# Function to calculate cumulative logits
model <- polr(NObeyesdad ~ fam_hist + FAF+NCP*FAVC*CAEC + TUE, 
              data = dat2, Hess=TRUE,method = "logistic")

sf <- function(probs) {
  sapply(1:(length(probs)-1), function(i) qlogis(sum(probs[1:i])))
}

# Get predicted probabilities for each level of the outcome
pred_probs <- predict(model, type = "probs")

# Calculate cumulative logits for each observation
logits1 <- t(apply(pred_probs, 1, sf))
par(mfrow = c(1,2))
# Plotting the cumulative logits against one predictor (e.g., Height)
# Assuming 'Height' is a continuous predictor in your model

## For height
plot(dat2$Height, logits1[,1], xlab = "Height", ylab = "Logit(Y>=1)",
     main = "Test of Proportional Odds Assumption")
points(dat2$Height, logits1[,2], col = "red")
points(dat2$Height, logits1[,3], col = "blue")

# Add legend to the plot
legend("topright", legend = c("Logit(Y>=1)", "Logit(Y>=2)", "Logit(Y>=3)"),
       col = c("black", "red", "blue"), pch = 1)


## For Age
logits2 <- t(apply(pred_probs, 1, sf))
plot(dat2$Age, logits2[,1], xlab = "Height", ylab = "Logit(Y>=1)",
     main = "Test of Proportional Odds Assumption")
points(dat2$Age, logits2[,2], col = "red")
points(dat2$Age, logits2[,3], col = "blue")

# Add legend to the plot
legend("topright", legend = c("Logit(Y>=1)", "Logit(Y>=2)", "Logit(Y>=3)"),
       col = c("black", "red", "blue"), pch = 1)


## testing the null of parallel regression 
tt = brant::brant(mod.up_reduced)










## (2) Residual Analysis: Conduct residual analysis to assess the model's goodness-of-fit. Calculate and inspect various residuals like deviance residuals, Pearson residuals, or scaled Schoenfeld residuals.
##______________________________________________________________________________
# # Get the number of levels in the outcome variable
# n_levels <- length(levels(dat2$NObeyesdad))
# 
# # Initialize a matrix to hold the deviance residuals for each class
# dev_resid <- matrix(NA, nrow = nrow(dat2), ncol = n_levels)
# 
# # Loop over each level of the outcome variable
# for (i in 1:n_levels) {
#   # Binarize the outcome variable at the current level
#   binary_outcome <- ifelse(dat2$NObeyesdad >= levels(dat2$NObeyesdad)[i], 1, 0)
#   
#   # Fit a logistic regression model for the binary outcome
#   binary_model <- glm(binary_outcome ~ Height + fam_hist + FAVC + NCP + 
#                         CAEC + CH2O + FAF + TUE + MTRANS, data = dat2, family = binomial())
#   
#   # Compute the deviance residuals for the binary model
#   dev_resid[,i] <- residuals(binary_model, type = "deviance")
# }
# 
# # Plot the deviance residuals for each class
# par(mfrow = c(2, 2))  # Adjust this to match the number of levels in your outcome variable
# for (i in 1:n_levels) {
#   plot(dev_resid[,i], main = paste("Deviance Residuals for Class", i),
#        xlab = "Observation Number", ylab = "Deviance Residuals")
#   abline(h = 0, lty = 2)
# }




## (3) Lack of Fit Test: Perform a formal test to check for lack of fit. One way to do this is by comparing the observed and predicted frequencies in each category using goodness-of-fit tests like the Pearson chi-squared test or likelihood ratio test.
##_____________________________________________________________________________
# model <- polr(formula = NObeyesdad ~ Height + fam_hist + FAVC + NCP + 
#                 CAEC + CH2O + FAF + TUE + MTRANS, data = dat2, Hess = TRUE, 
#               method = "logistic")

# Get the number of levels in the outcome variable
n_levels <- length(levels(dat2$NObeyesdad))
# Get predicted probabilities for each level of the outcome
pred_probs <- predict(model, type = "probs")
# Calculate observed probabilities for each level of the outcome
obs_probs <- table(dat2$NObeyesdad) / nrow(dat2)
# Plot observed vs fitted probabilities for each class
plot(1:n_levels, obs_probs, type = "b", pch = 19, xlab = "Class", ylab = "Probability",
     main = "Observed vs Fitted Probabilities", ylim = range(c(obs_probs, pred_probs)))
points(1:n_levels, colMeans(pred_probs), col = "red", type = "b", pch = 18)
legend("topright", legend = c("Observed", "Fitted"), col = c("black", "red"), pch = c(19, 18))


# Chi-squared test: The chi-squared test assesses whether a statistically significant difference exists between the observed and predicted frequencies. The null hypothesis of the chi-squared test is that there is no difference between the observed and predicted frequencies, implying that the model fits the data well.
obs_pred <- cbind(table(dat2$NObeyesdad), table(predict(model, type = "class")))
obs_pred <- as.data.frame(obs_pred)
colnames(obs_pred) <- c("Observed", "Predicted")
chisq.test(obs_pred)



# (4) Check for Influential Observations: Identify influential observations that may have a significant impact on the model's estimates by examining leverage, Cook's distance, or hat values.
##------------------------------------------------------------------------------
# Leverage calculation
modelg <- polr(formula = NObeyesdad ~ Height + fam_hist + FAVC + NCP +
                CAEC + CH2O + FAF + TUE + MTRANS, data = dat2, Hess = TRUE,
              method = "logistic")
# Initialize a matrix to hold the parameter estimates
params <- matrix(NA, nrow = nrow(dat2), ncol = length(coef(modelg)))
# Loop over each observation
for (i in 1:nrow(dat2)) {
  # Fit the model without the i-th observation
  model_i <- update(modelg, data = dat2[-i, ])
  # Store the parameter estimates
  params[i, ] <- coef(model_i)
}
# Compute the change in parameter estimates
change <- max(abs(params - coef(modelg)))
# Print the maximum change: This gives the maximum change in any parameter estimate when each observation is left out. Observations that cause a large change might be considered influential. A common rule of thumb is that an observation might be influential if removing it changes the coefficient estimate by more than 20%.
print(change) #1.6071

#(5) Variable Importance: look at the magnitude of the coefficients. Larger coefficients (in absolute value) indicate more important variables. Depending on the context of the analysis, the interpretation of the models may vary. For example, if one model ranks a certain predictor variable as highly influential while the other model ranks it lower, it could suggest that the relationship between that predictor and the response is not stable across different modeling approaches. inconsistent rankings of coefficients between models in sensitivity analysis highlight the need for cautious interpretation and further investigation to ensure robust and reliable model results. [Maybe talk about the variable selection approaches??]
##------------------------------------------------------------------------------
# Get the coefficients
coef <- coef(model)
# Get the absolute values of the coefficients
abs_coef <- abs(coef)
# Create a data frame for plotting
coef_df <- data.frame(Variable = names(abs_coef), Importance = abs_coef)
# Order the variables by importance
coef_df <- coef_df[order(coef_df$Importance, decreasing = TRUE), ]
aa = ggplot(coef_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Variable Importance\n (Logistic method)", x = "Variable", y = "Importance")
aa



# (6) Check for Multicollinearity: Values of vif up to 5 are usually interpreted as uncritical, values above 5 denote a considerable multicollinearity.
##------------------------------------------------------------------------------
# VIF calculation
vif <- car::vif(modelg)
print(vif)


# (7) Model Comparisons: lower is better
##------------------------------------------------------------------------------

# (8) Sensitivity Analysis --sensitivity analysis by assessing the robustness of your results to changes in modeling assumptions or data specifications. (Example: changed method argument in polr function).
##------------------------------------------------------------------------------
sensitivity_model <- polr(NObeyesdad ~ fam_hist + FAF+NCP*FAVC*CAEC + TUE, 
                          data = dat2, Hess=TRUE,method = "probit") # Changed method to probit
# do variable importance again and compare to the original by plotting side by side
coef <- coef(sensitivity_model)
# Get the absolute values of the coefficients
abs_coef <- abs(coef)
# Create a data frame for plotting
coef_df <- data.frame(Variable = names(abs_coef), Importance = abs_coef)
# Order the variables by importance
coef_df2 <- coef_df[order(coef_df$Importance, decreasing = TRUE), ]
bb = ggplot(coef_df2, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Variable Importance\n (Probit Method)", x = "Variable", y = "Importance")
bb

# check with random forest to see
set.seed(123)  # for reproducibility
model_rf <- randomForest(NObeyesdad ~ fam_hist + FAF+NCP*FAVC*CAEC + TUE, data = dat2, importance = TRUE)
importance <- importance(model_rf)
#for (class in colnames(importance)) {
imp_df <- data.frame(Variable = rownames(importance), Importance = importance[,class])
cc = ggplot(imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Variable Importance\n (Random Forest))",
       x = "Variable", y = "Importance")
#}
gridExtra::grid.arrange(aa, bb, cc, ncol = 3)
