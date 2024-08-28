
if("pacman" %in% installed.packages()[,1]){
  library(pacman)
}else{
  install.packages("pacman")
  library(pacman)
}

pacman::p_load(here, caret
               , corrplot
               , rms, ordinal, ordinalNet
               , psych
               , DescTools
               , readr
               , PRROC, pROC
)

    # specify input dataframe
    score <- # binary input
    ordinalNet_pred <- # prob
    rf_pred <- # prob
    xgb_pred <- # prob
    latte_pred <- # prob
    .dat_train <- data.frame(ordinalNet_pred=ordinalNet_pred,rf_pred=rf_pred,xgb_pred=xgb_pred,latte_pred=latte_pred,score = score)
    .dat_test <- # a dataframe similar to .dat_train without the score column
    ensemble_dir <- 
# Fit a logistic regression model

    selected_features <- c("score","ordinalNet_pred", "rf_pred", "xgb_pred", "latte_pred")
    ensemble_model <- glm(score ~ ., data = subset(.dat_train, select = selected_features), 
                 family = "binomial", maxit = 10000)
    saveRDS(ensemble_model, ensemble_dir)

# Pred on test data
    ensemble_model <- readRDS(here(ensemble_dir, paste0(.prefix, "ensemble_",.model,".RDS")))
    ensemble_pred <- predict(ensemble_model, newdata = subset(ensemble_test_subdf, select = selected_features[-1]), 
                           type = "response")
    print("Complete fitting ensemble model and predict on test set...the outputed dataset is named ensemble_pred")
    print(head(ensemble_pred))