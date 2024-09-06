#### load packges and functions ####
library(MASS)
library(pROC)
library(dplyr)
#==================================================================================
# n: number of observations
# p: number of features

##### Step1_generateX #####
# @family: the type of X-features
# c(rep("binary",4), rep("count",4), rep("numeric",2)) indicates 4 binary, 4 counting and 2 numeric covariates
# @rho: correlation matrix of MVN = (1-rho)I + rho
# @coef: coefs for transforming MVN to Binary or Numeric
#----------------------------------------------------------
generate_X = function(n, p = 10, family = c(rep("binary",4), rep("count",4), rep("numeric",2)),
                      rho = 0.1, coef = c(0.3,2)){
  # generate MVN
  Sigma = (1-rho)*diag(1,nrow = p) + rho*matrix(1, p, p)
  X = mvrnorm(n = n, rep(0,10), Sigma)
  # convert to different types
  for (j in 1:p) {
    if(family[j] == 'binary'){
      X[,j] = ifelse(X[,j]>coef[1],1,0)
    }
    
    if(family[j] == 'count'){
      X[,j] = sapply(X[,j],FUN = function(x){
        rpois(1,lambda = coef[2]*pnorm(x))})
    }
    
    if(family[j] == 'numeric'){
      X[,j] = pnorm(X[,j])
    }
  }
  
  X.scaled = scale(X,scale = FALSE)
  return(X.scaled)
}

##### Step2_generateT #####
# @X: feature matrix
# @b0,beta: coefficients of the linear components in the model
# @coef: coefs for cox model
# --------------------------------------------------------------
generate_T = function(X, b0 = -3,
                      beta = 0.5 * c(1, -1, -2, -2, 1, -1, -2, -2, 1, -2),
                      coef = 0.6)
{

  T = apply(X, 1, FUN = function(X_i){
      rexp(1, exp(t(X_i)%*%beta+b0)*coef)
    })

  return(T)
}


##### Step3_generateC #####
# @T: Event time vector
# @family: hazard model of event time, "linear", "quadratic"
# @t_max,c_min: the time period when a censor might happen 
# --------------------------------------------------------------
generate_C = function(n, T)
{

  t_max = 23
  c_min = 20
  C = runif(n, c_min, t_max+1) 
  
  return(C)
}

##### Step4_generateData #####
# @T.family: 'family' argument for generate_T
# @T.b0, T.beta, T.B: coefficients argument for generate_T
# @C.rate: 'rate' argument for generate_C(not needed in this version)
# @filename: data matrix for simulation, '.csv' file will be saved in working dir
# ----------------------------------------------------------------------
gen_data = function(n, p=10,
                    T.b0, T.beta, T.coef,
                    C.rate,a,b,c,
                    filename,dataname)
{
  ## Generate X Variables
  X = generate_X(n,p)
  ## Generate T Variables Based on Classical Cox Proportional Hazards Models
  T = generate_T(X=X,b0 = T.b0, beta = T.beta,coef = T.coef)
  ## Generate Censor Time
  C = generate_C(n, T)

  I = ifelse(T<=C,1,0)

  gen_single_patient = function(i){
    Censor = floor(C[i])
    T_star = floor(T[i])
    ID = rep(i,Censor+1)
    if(T_star < Censor){Y = c(rep(0,T_star),rep(1,Censor-T_star+1))}
    if(T_star >= Censor){Y = rep(0,Censor+1)}
    t = 0:Censor
    

    H_t = rpois(length(t), lambda = ifelse(t < T_star, a[1], ifelse(t >= T_star + 2, c[1], b[1])))
    W1_t = rpois(length(t), lambda = ifelse(t < T_star, a[2], ifelse(t >= T_star + 2, c[2], b[2])) * H_t)
    W2_t = rpois(length(t), lambda = ifelse(t < T_star, a[3], ifelse(t >= T_star + 2, c[3], b[3])) * H_t)
    W3_t = rpois(length(t), lambda = ifelse(t < T_star, a[4], ifelse(t >= T_star + 2, c[4], b[4])) * H_t)
    W4_t = rpois(length(t), lambda = ifelse(t < T_star, a[5], ifelse(t >= T_star + 2, c[5], b[5])) * H_t)
    W5_t = rpois(length(t), lambda = ifelse(t < T_star, a[6], ifelse(t >= T_star + 2, c[6], b[6])) * H_t)
    
    ## Generate Surrogates
    S_t = sapply(t, function(ti) {
      if (ti < T_star) {
        rbeta(1, 1, 3)
      } else {
        rbeta(1, 1 + (ti - T_star)/2, 1)
      }
    })

    X_feature = matrix(X[i,],1,p)[rep(1,Censor+1),]
    colnames(X_feature) = paste0("X", 1:p)
    

    single_patient = data.frame(bool_data = rep(TRUE, length(t)),silver = S_t,T = t, Y = Y,ID = ID, W1 = W1_t, W2 = W2_t, 
                                W3 = W3_t, W4 = W4_t, W5 = W5_t)
    single_patient = cbind(single_patient, X_feature)
    
    if (nrow(single_patient) < 24) {
      num_missing_rows = 24 - nrow(single_patient)
      additional_rows = data.frame(
        bool_data = rep(FALSE, num_missing_rows),
        silver = rep(0, num_missing_rows),
        T = seq(max(single_patient$T) + 1, max(single_patient$T) + num_missing_rows),  # 递增的时间点
        Y = rep(0, num_missing_rows),
        ID = rep(i, num_missing_rows), 
        W1 = rep(0, num_missing_rows),
        W2 = rep(0, num_missing_rows),
        W3 = rep(0, num_missing_rows),
        W4 = rep(0, num_missing_rows),
        W5 = rep(0, num_missing_rows)
      )

      additional_X_feature = matrix(0, num_missing_rows, p)
      colnames(additional_X_feature) = colnames(X_feature)
      single_patient = rbind(single_patient, cbind(additional_rows, additional_X_feature))
    }
    
    single_patient = as.matrix(single_patient)
    
    return(single_patient)
  }

  data = data.frame()
  for (i in 1:n) {
    data = rbind(data,gen_single_patient(i))

    # monitor
    if (i %% 1000 == 0){
        print(paste0(i/n*100,'%'))
    }

  }
  write.csv(data, file = filename)
}


# Generate Embeddings Files 
gen_emb <- function(train_file,test_file,embeddings_file)
{
  train_data = read.csv(train_file)
  test_data = read.csv(test_file)
  
  data = rbind(train_data,test_data)
  
  W_matrix = data[,7:11]
  num_W_vars = 5
  
  cooccurrence_matrix <- matrix(0, num_W_vars, num_W_vars)
  
  ## Calculate The Co-occ Matrix
  for (i in 1:nrow(W_matrix)) {
    for (j in 1:num_W_vars) {
      for (k in j:num_W_vars) {
        if (W_matrix[i, j] > 0 && W_matrix[i, k] > 0) {
          cooccurrence_matrix[j, k] <- cooccurrence_matrix[j, k] + 1
          if (j != k) {
            cooccurrence_matrix[k, j] <- cooccurrence_matrix[k, j] + 1
          }
        }
      }
    }
  }
  
  SPPMI_matrix = calculate_SPPMI(cooccurrence_matrix,k=1,num_W_vars=num_W_vars)
  
  embeddings = obtain_embeddings(SPPMI_matrix)
  
  rownames(embeddings) = paste0("W",1:5)
  
  write.csv(embeddings, file = embeddings_file)
}



# Calculate SPPMI Matrix
calculate_SPPMI <- function(cooccurrence_matrix, k = 1,num_W_vars) 
{

  total_occurrences <- sum(cooccurrence_matrix)
  k <- 1  # Neg Sampling Parameter
  
  SPPMI_matrix <- matrix(0, num_W_vars, num_W_vars)
  
  for (i in 1:num_W_vars) {
    for (j in 1:num_W_vars) {
      if (cooccurrence_matrix[i, j] > 0) {
        pmi_value <- log((cooccurrence_matrix[i, j] * total_occurrences) /
                           (sum(cooccurrence_matrix[i, ]) * sum(cooccurrence_matrix[, j]))) - log(k)
        SPPMI_matrix[i, j] <- max(pmi_value, 0)
      }
    }
  }
  
  return(SPPMI_matrix)
}


obtain_embeddings <- function(SPPMI_matrix)
{

  pca_result <- prcomp(SPPMI_matrix, scale. = TRUE)

  embeddings <- pca_result$x
  
  return(embeddings)
}

gen_fts <- function(fts_file)
{
  if(grepl("W", fts_file) | grepl("w", fts_file))
  {
    data = data.frame(x=paste0("W",1:5))
  } else{
    data = data.frame(x=paste0("X",1:10))
  }
  write.csv(data,file = fts_file)
}







