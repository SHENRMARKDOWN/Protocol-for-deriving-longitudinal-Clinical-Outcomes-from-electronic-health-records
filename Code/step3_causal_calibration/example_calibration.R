# Modified based on Dominic Disanto's code
# Amiee Huang

gaus4 = function(x) 0.5*(3-x^2)*dnorm(x)

############################## # 
# expit and adalasso 
############################## #

expit = function(x)
{
  return(1/(1+exp(-x)))
}

# explicit adaptive lasso, useful for debugging 
adalasso.explicit = function(x, y, family
                             , type.measure
                             , nfolds
                             , lambda
                             , lambda.min, .return_model=F)
{
  ridgefit = glmnet::cv.glmnet(alpha = 0, ...)
  adawgt = 1/abs(coef(ridgefit, s = "lambda.min")[-1])
  if(missing(lambda.min))
  {
    adafit = glmnet::cv.glmnet(penalty.factor = adawgt, ...)
    lambda.min = adafit$lambda.min
  }else
  {
    adafit = glmnet(penalty.factor = adawgt, ...)
  }
  if(.return_model){
    return(adafit)
  }else{
    return(drop(coef(adafit, s = lambda.min)))
  }
}


# A simple adaptive lasso
adalasso = function(..., lambda.min, .return_model=F)
{
  ridgefit = glmnet::cv.glmnet(alpha = 0, ...)
  adawgt = 1/abs(coef(ridgefit, s = "lambda.min")[-1])
  if(missing(lambda.min))
  {
    adafit = glmnet::cv.glmnet(penalty.factor = adawgt, ...)
    lambda.min = adafit$lambda.min
  }else
  {
    adafit = glmnet(penalty.factor = adawgt, ...)
  }
  if(.return_model){
    return(adafit)
  }else{
    return(drop(coef(adafit, s = lambda.min)))
  }
}


############################## #
# ordinal model coefficients
############################## #

ordinalCoefs= function(...
                       , .link_fn="logit", .alpha=1, .nlambda=20, .nFolds=5
                       ){
  
  # ordinalNet fitting 
  .oNet <- ordinalNet(#... # x,y sufficient
                                    # ,
                                   x
                                   , factor(y)
                                   , alpha = .alpha
                                   , nLambda = .nlambda
                                   , family = "cumulative"
                                   , link = .link_fn
                                   # , nFolds = .nFolds # needed for ordinalNetTune
                                   )
  
  # coefficient extraction 
  .oNet_beta = coef(.oNet
                    # , whichLambda = which.min(apply(.oNet$misclass, 1, mean))
                    )
  
  return(.oNet_beta)
}


############################## #
# kernel smoothed pi 
############################## #

PS_kernel=function(Xi, Ti, Yi, .family, .outcome, .nfolds=5, weights=NULL){
  
  if(is.data.frame(Xi)) Xi = as.matrix(Xi) else if(!is.matrix(Xi)) stop("Xi not matrix or dataframe cannot continue")
  
  if(is.null(weights)) weights = rep(1, length(Ti))
  # treatment ALASSO (Cheng eqn 9)
  Trt.coef = adalasso(x = Xi
                      , y = Ti
                      , family = "binomial" 
                      , nfolds=.nfolds
                      , weight = weights
  )
  
  
  # outcome ALASSO (Cheng eqn 10) 
  OC.coef = adalasso(x = cbind(Xi[!is.na(Yi),]
                               , Trt=Ti[!is.na(Yi)]
  )
  , y = Yi[!is.na(Yi)]
  , family = .family
  , nfolds = .nfolds
  , weights = weights[!is.na(Yi)]
  , ridge.penalty = c(rep(1, ncol(Xi)), 0)
  , .penalize_Trt = F
  # , .return_model = T
  )
    
  # Kernel estimator based on coefs from 9/10 above 
  a = cbind(rep(1,nrow(Xi)), Xi) %*% t(t(Trt.coef))
  
  OC.coef.noTrt = OC.coef[-which(names(OC.coef)=="Trt")]
  b = cbind(rep(1,nrow(Xi)), Xi) %*% t(t(OC.coef.noTrt))
  S = cbind(a, b)
  
  S_preproc = apply(S, 2, function(x) (x-mean(x))/ifelse(sd(x)==0, 1, sd(x)) )
  S_cdf = apply(S_preproc, 2, pnorm)
  
  plugin = c(sd(S_cdf[1,])*nrow(Xi)^(-1/5)
             , sd(S_cdf[2,])*sum(!is.na(Yi))^(-1/5)
  )
  
  np_preproc_pred = c()
  for(i in 1:nrow(S_cdf)){
    num1 = gaus4((-S_cdf[,1] + S_cdf[i,1]) / plugin[1])
    num2 = gaus4((-S_cdf[,2] + S_cdf[i,2]) / plugin[2])
    np_preproc_pred = c(np_preproc_pred, (t(num1*num2) %*% (Ti * weights)) / t(num1 * num2) %*% weights)
  }
  
  
  return(list(PS.Kernel = np_preproc_pred
              , Trt_IntOnly = as.logical(sd(S[1,])==0)
              , Outcome_IntOnly = as.logical(sd(S[2,])==0)
              , Trt.coef = Trt.coef
              , OC.coef = OC.coef
  )
  )
  
}


############## # 
# Robust ATE 
############## #

ATE_DiPS = function(Ti, Yi, PS, .IPW_stabilize, min.ipw=0.1, max.ipw=10
                    , z_star
                    , weights = NULL
                    , crump.trim = T
                    , labels){
  
  if(!is.null(weights)) {
    if(length(weights)!=length(Ti)) stop("Weights supplied but incorrect length, should match Ti vector length")
  }
  trimmed = F 
  
  if(crump.trim){
    .labeled.removed = sum((PS<0.05 | PS>0.95) & !is.na(labels) )
    .unlabeled.removed = sum((PS<0.05 | PS>0.95) & is.na(labels) )
    if(.labeled.removed + .unlabeled.removed <= 0.4*length(labels)){
      trimmed = T
      .keep = which(PS>=0.05 & PS<=0.95)
      if(length(.keep) != 0){
        PS = PS[.keep]
        weights = weights[.keep]
        Ti = Ti[.keep]
        Yi = Yi[.keep]
      }else{
        print("Error: nothing to keep!")
      }
    }
  }else{
    .labeled.removed = 0
    .unlabeled.removed = 0
    .n.removed = 0
  }
  
  IPW = Ti/PS + (1-Ti)/(1-PS)
  
  mu_1 = sum(weights*Ti*Yi / PS) / sum(Ti*weights / PS)
  mu_0 = sum(weights * (1-Ti)* Yi / (1-PS)) / sum((1-Ti) * weights / (1-PS))

  .output = list(.ate = mu_1 - mu_0
                 , .labeled.removed = .labeled.removed
                 , .unlabeled.removed = .unlabeled.removed
                 , trimmed = trimmed
                 #, .keep = .keep
                )
  return(.output)
  
}

############### #
# Full calculation
############### # 
AIPW_robust_analytic_perturb = function(analytic_object
                                , .analytic_object_1yr # misnomer, can take any lookforward, I just didn't bother to rename and ensure correctness
                                , .outcome 
                                , pweights = NULL
                                , .lookforward_window = NULL 
                                , cv.outcome = T
                                , preDMT_imputed_score = NULL
                                , postDMT_imputed_score = NULL
                                , cutoff = 0
                                , imputed_score_as_confounding = F
                                , add_transform_var = F
                                , Trt.model = "adalasso" # adalasso, rf, xgb
                                ){
    
  if(is.null(.lookforward_window)) {
    warning("Setting lookforward to 1 year by default")
    .lookforward_window = dyears(1)
  }
  
  ## Kernel Smoothing #### 
  
  if(imputed_score_as_confounding){
    X_preDMT_df <- cbind(analytic_object$X, preDMT_imputed_score)
  }else{
    X_preDMT_df <- analytic_object$X
  }
 
  X_preDMT = as.matrix(X_preDMT_df)
  
  propens = PS_kernel(Xi = X_preDMT
                      , Ti = analytic_object$A
                      , Yi = analytic_object[[.outcome]]
                      , .outcome = .outcome
                      , .nfolds=5
                      , weights = pweights
                      )
  
  
  ## Utility Covariate (truncating/cleaning) for robust imputation (Cheng 2021)
  .pi = propens$PS.Kernel
  .pi_trim = pmax(pmin(.pi, 0.99), 0.01)
  U_pi = analytic_object$A/.pi_trim + (1-analytic_object$A)/(1-.pi_trim)

  ## Combining 
  Trt = analytic_object$A

  Xfull = cbind(
    X_preDMT
    , score = postDMT_imputed_score
    , Trt 
    , U_pi 
    )
  
  
  
  ## Outcome imputation #### 
  if(cv.outcome){
    outcome_indx = which(!is.na(analytic_object[[.outcome]]))
    # simple sampling 
    cv_outcome_indx = sample(outcome_indx, replace=F, size = 0.8*length(outcome_indx))
    test_outcome_indx = setdiff(outcome_indx, cv_outcome_indx)

    # adalasso 
    Xi_Outcome_Coefs = adalasso(x = as.matrix(Xfull[cv_outcome_indx,])
                                , y = analytic_object[[.outcome]][cv_outcome_indx]
                                , family = .family
                                , type.measure = ifelse(.family=="binomial", "auc", "mse") # "auc"
                                , nfolds = 5
                                , .return_model=T
                                , weights = pweights[cv_outcome_indx])
    
    
    Y_test_class = predict(Xi_Outcome_Coefs, newx = as.matrix(Xfull[test_outcome_indx,]), s="lambda.min", type = "class")
    Y_test_prob = predict(Xi_Outcome_Coefs, newx = as.matrix(Xfull[test_outcome_indx,]), s="lambda.min", type = "response")
    test_acc = sum(Y_test_class==analytic_object[[.outcome]][test_outcome_indx]) / length(test_outcome_indx)
    test_auc = survival::concordancefit(y = analytic_object[[.outcome]][test_outcome_indx]
                                        , x = Y_test_prob)$concordance
    
  }else{
    test_acc = test_auc = NULL
  }
  
  # Fitting models on full data 
    # adalasso 
  Xi_Outcome_Coefs = adalasso(x = as.matrix(Xfull[which(!is.na(analytic_object[[.outcome]])),])
                              , y = analytic_object[[.outcome]][which(!is.na(analytic_object[[.outcome]]))]
                              , family = .family
                              , type.measure = ifelse(.family=="binomial", "auc", "mse") # "auc"
                              , nfolds = 5
                              , .return_model=T
                              , weights = pweights[which(!is.na(analytic_object[[.outcome]]))])
    
  
  Y_pred = predict(Xi_Outcome_Coefs, newx = as.matrix(Xfull), s="lambda.min", type = "class")                  
  Y_dag = coalesce(analytic_object[[.outcome]], as.integer(Y_pred))
  # check, should be TRUE 
  # all.equal(Y_dag[!is.na(analytic_object[[.outcome]])], analytic_object[[.outcome]][!is.na(analytic_object[[.outcome]])])
  
  
  ## ATE Estimation ####
  
  .dips.output = ATE_DiPS(Ti = analytic_object$A
                          , Yi = Y_dag
                          , PS = .pi_trim
                          , min.ipw = 0.01 
                          , max.ipw = 10
                          , .IPW_stabilize = T
                          , weights = pweights
                          , labels = analytic_object[[.outcome]])
  
  
  bcd_ntz_ate = .dips.output$.ate 
  
  
  if(cv.outcome){
    return(list(ATE = bcd_ntz_ate
                , Impute_Acc = test_acc
                , Impute_AUC = test_auc
                , Crump.Labeled.Removed = .dips.output$.labeled.removed
                , Crump.Unlabeled.Removed = .dips.output$.unlabeled.removed
                , Crump.Trimmed = .dips.output$trimmed
                , Trt_KS_IntOnly = propens$Trt_IntOnly
                , Outcome_KS_IntOnly = propens$Outcome_IntOnly
                , OC.coef = propens$OC.coef
                , Trt.coef = propens$Trt.coef
                , DiPS.notrim = propens$PS.Kernel
                , Y_dag = Y_dag
                , ImputeModel = list(Model = Xi_Outcome_Coefs
                                     , data = Xfull 
                                     , outcome = analytic_object[[.outcome]]
                                     , cv_outcome_indx = cv_outcome_indx
                                     , test_outcome_indx = test_outcome_indx
                )
    )
    )
  }else{
    return(list(ATE = bcd_ntz_ate
                , Trt_KS_IntOnly = propens$Trt_IntOnly
                , Outcome_KS_IntOnly = propens$Outcome_IntOnly
                , Crump.Labeled.Removed = .dips.output$.labeled.removed
                , Crump.Unlabeled.Removed = .dips.output$.unlabeled.removed
                , Crump.Trimmed = .dips.output$trimmed
                , OC.coef = propens$OC.coef
                , Trt.coef = propens$Trt.coef
                , DiPS.notrim = propens$PS.Kernel
    )
    )
  }

}