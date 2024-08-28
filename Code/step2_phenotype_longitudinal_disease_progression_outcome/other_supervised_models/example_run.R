### 
### Amiee_Huang

if("pacman" %in% installed.packages()[,1]){
  library(pacman)
}else{
  install.packages("pacman")
  library(pacman)
}

pacman::p_load(dplyr, magrittr, stringr, lubridate, tidyr# data cleaning
               , arrow
               , rms, ordinal, ordinalNet
               , readr, readxl, here, openxlsx #import/export
               , glmnet # analysis
               , ranger, randomForest,xgboost # RF
               , caret
               , corrplot
               , psych
               , DescTools
               , PRROC, pROC
               , update=F)    

select <- dplyr::select # Resolve conflict with MASS ridge function 
summarize <- dplyr::summarize # Resolve conflict with Hmisc  
union = dplyr::union # silent conflict with lubridate 


`%nin%` <- Negate(`%in%`)

# input & output dir
input_dir = #
ordinalNet_dir = #
rf_dir = #
xgb_dir = #

# data loading
df_train <- read.csv(file.path(input_dir, "df_train.csv"))
df_test <- read.csv(file.path(input_dir, "df_test.csv"))
fts <- read.csv(file.path(input_dir,"df_fts.csv"))$x

.x <- df_train  %>% 
            select(-c(X,T,Y,ID)) %>%
            mutate(across(.cols = !!fts, ~ log(.x+1))) %>%
            mutate_all(~ .x /max(1,(max(.x) - min(.x)))) %>% as.matrix()

# discretization & cleaning: suppose that 4 is the cutoff threshold        
.y <- df_train$Y[!is.na(df_train$Y)]
.y[.y < 4] <- 0
.y[.y >= 4] <- 1
score = as.factor(.y)


# M1: LASSO
ordinalNet <- ordinalNetTune(x = .x
                                , y = score
                                , alpha = 1
                                , family = "cumulative"
                                , link = "logit"
                                , nFolds = 10
)
saveRDS(ordinalNet, ordinalNet_dir)

# M2: RF

rf <- randomForest(score ~ .,
                      data = .x,
                      importance = TRUE,
                      proximity = TRUE)

saveRDS(rf, rf_dir)

# M3: XGB:
.x_xgb <- list("data" = as.matrix(.x),
                                "label" = .y)
xgb <- xgboost(data = .x_xgb$data
                  , label = .x_xgb$label
                  , max.depth = 200
                  , eta = 1
                  , nthread = 1
                  , nrounds = 500
                  , objective = "binary:logistic"
)
saveRDS(xgb, xgb_dir)


# TEST

.test_x <- df_test %>% 
  select(-c(X,T,Y,ID)) %>%
  mutate(across(.cols = !!fts, ~ log(.x+1))) %>%
  mutate_all(~ .x /max(1,(max(.x) - min(.x)))) %>% as.matrix()

.test_y <- df_test$Y[!is.na(df_test$Y)]
.test_y[.test_y < 4] <- 0
.test_y[.test_y >= 4] <- 1
score_test = as.factor(.test_y)

.test_x_xgb <- list("data" = as.matrix(.test_x),
                    "label" = .test_y)

ordinalNet_pred = predict(ordinalNet$fit
                             , newx = .test_x
                             , type = "response"
                             , whichLambda = which.min(apply(ordinalNet$misclass, 1, mean)))[,2]

rf_pred <- predict(rf
                      , newdata = .test_x
                      , type = "prob")[,2]

xgb_pred <- predict(xgb
                       , newdata = .test_x_xgb$data
                       , type = "prob")
