## Set WD
options("install.lock"=FALSE)

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

## Missing data
vis_miss(dat, warn_large_data = F) #no missing data

## Data structure
str(dat)

## Outcome Variable
ggplot(dat, aes(x = NObeyesdad))+ 
  geom_bar(fill = "grey", color = "black", alpha = 1)+
  xlab("Obesity Status")


## Summary Statistics
dat %>% tbl_summary(digits = list(everything() ~ c(2)),
                    statistic = list(all_continuous() ~ "{mean} ({sd})"),
                    by = NObeyesdad,
                    missing = "no",
                    label = list(
                      family_history_with_overweight ~"Family history overweight", 
                      FAVC ~ "High-calorie food consumption",
                      FCVC ~ "Number of meals veggie consumption",
                      NCP ~ "Daily main meals",
                      CAEC ~ "Eating between meals",
                      SMOKE ~ "Smoke",
                      CH2O ~ "Water Intake (L)",
                      SCC ~ "Calorie intake monitoring",
                      FAF ~ "Frequency of days of physical activity (wk)",
                      TUE ~ "Technology use time",
                      CALC ~ "Frequency of alchohol intake",
                      MTRANS ~ "Transportation"))%>%
  add_overall() %>% 
  add_n() %>%
  bold_labels()

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
table(dat$NObeyesdad)

## Factor variables
dat <- dat %>%
  mutate_if(is.character, as.factor)

## Sumary Statistics
dat %>% tbl_summary(digits = list(everything() ~ c(2)),
                    statistic = list(all_continuous() ~ "{mean} ({sd})"),
                    by = NObeyesdad,
                    missing = "no",
                    label = list(
                      family_history_with_overweight ~"Family history overweight", 
                      FAVC ~ "High-calorie food consumption",
                      FCVC ~ "Number of meals veggie consumption",
                      NCP ~ "Daily main meals",
                      CAEC ~ "Eating between meals",
                      SMOKE ~ "Smoke",
                      CH2O ~ "Water Intake (L)",
                      SCC ~ "Calorie intake monitoring",
                      FAF ~ "Frequency of days of physical activity (wk)",
                      TUE ~ "Technology use time",
                      CALC ~ "Frequency of alchohol intake",
                      MTRANS ~ "Transportation"))%>%
  add_overall() %>% 
  add_n() %>%
  bold_labels()


## Some plots (not usefull?)
my_palette <- c("darkblue", "darkgreen", "darkred", "purple4")

ggplot(dat, aes(x = MTRANS, fill = NObeyesdad)) +
  geom_bar(position = "dodge", color = "black") +
  scale_fill_manual(values = my_palette, name = "Obesity Status", 
                    labels = c("Normal", "Obese", "Overweight", "Underweight")) +
  labs(x = "Transportation", y = "Count", fill = "Obesity Status") +
  ggtitle("Obesity Status by Vehicle Used")

df <- as.data.frame(prop.table(table(dat$FAVC,dat$NObeyesdad),margin = 2))
df$Freq <- df$Freq*100
df <- df %>% rename(FAVC = "Var1",
                    NObeyesdad = "Var2")

ggplot(dat, aes(x = FAVC, fill = NObeyesdad)) +
  geom_bar(position = "dodge", color = "black") +
  scale_fill_manual(values = my_palette, name = "Obesity Status", 
                    labels = c("Normal", "Obese", "Overweight", "Underweight")) +
  labs(x = "High Caloric Food Consumption", y = "Count", fill = "Obesity Status") +
  ggtitle("Obesity Status by Vehicle Used") +
  geom_text(data = df, aes(label = paste0(round(Freq, 1), "%"),
                           y = Freq, group = NObeyesdad),
            position = position_dodge(width = 0.9), size = 3, color = "black")



## Correlation
dat %>% select_if(is.numeric) %>% cor() %>% corrplot()
#Height and weight positively corr
#Age and tech negatively corr
#Age and weight positively corr


## Factor more variables 
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

# Fit a cumulative logistic regression model
mod_polr <- polr(NObeyesdad ~., 
                 data = dat, Hess=TRUE,method = "logistic")
## Summary
summary(mod_polr)
stargazer(mod_polr, type="text", style="apsr", single.row = T)



## Plot Estimated slopes
plot_ic <- function(data, title) {
  data <- data %>% mutate(estimate = (`2.5 %` + `97.5 %`)/2)
  data$vars <- rownames(data)
  
  ggplot(data, aes(x = estimate, y = vars)) +
    geom_point() +
    geom_segment(aes(x = `2.5 %`, xend = `97.5 %`, yend = vars)) +
    geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
    ggtitle(title)
}
confints <- confint(mod_polr)
plot_ic(as.data.frame(confints),"95% CI") ## Too many variables 


## Get predictions
predictions <- predict(mod_polr, newdata = dat)
confusionMatrix(predictions,dat$NObeyesdad) #Very bad model

## Too many variables. Variable selection by Lasso
## Set model Matrix
X <- model.matrix(NObeyesdad ~., 
                  data = dat,
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
Y <- dat$NObeyesdad

## Fit proportional odds model with lasso penalty
lasso.mod <- ordinalNet(X, Y, 
                        family="cumulative",
                        link="logit",
                        alpha = 1)

## Coefficients
coef(lasso.mod, matrix = T) #age,heigt NCP,CH2O,FAF,TUE


## Fit reduced model
mod_polr_reduced <- polr(NObeyesdad ~ Age + Height + NCP + CH2O + FAF +TUE, 
                         data = dat, Hess=TRUE,method = "logistic")
summary(mod_polr_reduced)

## Coefficients
confints <- confint(mod_polr_reduced)
plot_ic(as.data.frame(confints),"95% CI")


## Try interactions
mod.amos1 <- polr(NObeyesdad ~ Age*Height +TUE + NCP+FAF + CH2O, 
                  data = dat, Hess=TRUE,method = "logistic")
stargazer(mod.amos1, type="text", style="apsr", single.row = T)
mod1 <- polr(NObeyesdad ~ Age+TUE+ Height + NCP*FAF + CH2O, 
             data = dat, Hess=TRUE,method = "logistic")
stargazer(mod1, type="text", style="apsr", single.row = T)

mod2 <- polr(NObeyesdad ~ Age+TUE+ Height + NCP+ FAF*CH2O, 
             data = dat, Hess=TRUE,method = "logistic")
stargazer(mod2, type="text", style="apsr", single.row = T)

mod3 <- polr(NObeyesdad ~ Age+ Height + NCP*FAF*CH2O + TUE, 
             data = dat, Hess=TRUE,method = "logistic")
stargazer(mod3, type="text", style="apsr", single.row = T)



## Model Comparisons (the reduced model with no interactions is prefered)
anova(mod_polr_reduced,mod.amos1)
anova(mod_polr_reduced,mod1)
anova(mod_polr_reduced,mod2)
anova(mod_polr_reduced,mod3) # Best model is the one with no interactions

AIC(mod_polr_reduced,mod1,mod2,mod3)
BIC(mod_polr_reduced,mod1,mod2,mod3)


## Predictions
predictions <- predict(mod_polr_reduced, newdata = dat)
confusionMatrix(predictions,dat$NObeyesdad) #Still very bad we try oversampling

## Oversampling
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

## Coefficients
coef(lasso.mod2, matrix = T) #all variables without Gender

## Predictions
predictions <- predict(lasso.mod2, newdata = dat2,type = "class")
confusionMatrix(as.factor(predictions),Y) #Class 1=Normal 2=Obese 3=Overweight 4=Underweigt


## Prop odds model With variables selected by LASSO (remove only Gender)
mod.up <- polr(NObeyesdad ~., 
               data = dat2 %>% dplyr::select(-Gender), Hess=TRUE,method = "logistic")
summary(mod.up)
stargazer(mod.up, type="text", style="apsr", single.row = T)

## Fit model with variables selected by P-value
mod.up_reduced <-  polr(NObeyesdad ~ Height + fam_hist+FAVC + NCP + CAEC + CH2O + FAF + TUE + MTRANS, 
                        data = dat2, Hess=TRUE,method = "logistic")
summary(mod.up_reduced)


## Fit Model with variables selected first (with data not upsampled)
mod.up_reduced2 <- polr(NObeyesdad ~ Age + Height + NCP + CH2O + FAF +TUE, 
                        data = dat2, Hess=TRUE,method = "logistic")


## Predictions
predictions <- predict(mod.up_reduced2, newdata = dat2)
confusionMatrix(predictions,dat2$NObeyesdad)

anova(mod.up,mod.up_reduced2)
anova(mod.up_reduced, mod.up) #model diagnostics on the mod.up_reduced (pvalue selection on upsampling data)





################################################################################
########################## Not important below #################################
################################################################################


## Cross Validation 
grid = seq(.005,0.05,length = 30)
lasso.mod2 <- ordinalNetCV(X, Y, 
                           family="cumulative",
                           link="logit",
                           alpha = 1,
                           lambdaVals = grid)

lasso.mod2$bestLambdaIndex
lasso.mod2$lambdaVals[lasso.mod2$bestLambdaIndex]

#0.013598702
lasso.mod2.cv <- ordinalNet(X, Y,
                            family="cumulative",
                            link="logit",
                            alpha = 1,
                            lambdaVals = 0.04689655)
coef(lasso.mod2.cv,matrix = T)     

## Predictions
predictions <- predict(lasso.mod2.cv, newdata = dat2,type = "class")
confusionMatrix(factor(predictions,levels = c("1","2","3","4")),Y) #Class 1=Normal 2=Obese 3=Overweight 4=Underweigt

mod.up <- polr(NObeyesdad ~ Age+fam_hist+FCVC+NCP+CAEC+FAF+TUE+MTRANS, 
               data = dat2, Hess=TRUE,method = "logistic")
summary(mod.up)
## Predictions
predictions <- predict(mod.up, newdata = dat2)
confusionMatrix(predictions,dat2$NObeyesdad)



## Neural Network
library(tidyverse)
library(neuralnet)
dat_subset <- dat %>% dplyr::select(Age,fam_hist,FCVC,NCP,CAEC,FAF,TUE,MTRANS,NObeyesdad)
dat_subset$fam_hist <- as.numeric(dat_subset$fam_hist)-1
dat_subset$FCVC <- as.numeric(dat_subset$FCVC)

dat_subset$NCP <- factor(dat_subset$NCP, levels = c("1","2","3","4"))
dat_subset$NCP <- as.numeric(dat_subset$NCP)

dat_subset$CAEC <- as.numeric(dat_subset$CAEC)

dat_subset$FAF <- as.numeric(dat_subset$FAF)-1
dat_subset$TUE <-as.numeric(dat_subset$TUE)-1
dat_subset$MTRANS <- as.numeric(dat_subset$MTRANS)


model <- neuralnet(
  NObeyesdad~.,
  data=dat_subset,
  hidden=1,
  linear.output = FALSE
)
