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

## Function to Plot IC
plot_ic <- function(data, title) {
  data <- data %>% mutate(estimate = (`2.5 %` + `97.5 %`)/2)
  data$vars <- rownames(data)
  
  ggplot(data, aes(x = estimate, y = vars)) +
    geom_point() +
    geom_segment(aes(x = `2.5 %`, xend = `97.5 %`, yend = vars)) +
    geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
    ggtitle(title)
}

## Variable Selection by Lasso = pvalue in this case
## Fit reduced model
mod_polr_reduced <- polr(NObeyesdad ~ Age + Height + NCP + CH2O + FAF +TUE, 
                         data = dat, Hess=TRUE,method = "logistic")
summary(mod_polr_reduced) 
stargazer(mod_polr_reduced, type="text", style="apsr", single.row = T)

confints <- confint(mod_polr_reduced)
plot_ic(as.data.frame(confints),"95% CI") ## Too many variables 

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

## Model Diagnostics on mod.up_int5









