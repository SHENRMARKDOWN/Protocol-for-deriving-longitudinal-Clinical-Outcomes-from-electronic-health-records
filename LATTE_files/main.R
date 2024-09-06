library(MAP)
library(ff)
library(data.table)
library(MASS)
source("simulation_data_functions_v4.R")



set.seed(123)  #
train_num = 1000
test_num = 500
train_file = "train.csv"
test_file = "test.csv"
embeddings_file = "embeddings.csv"
w_fts = "w_fts.csv"
x_fts = "x_fts.csv"

#----------------------------------------
# data generation #
#----------------------------------------



#Generate Parameter a,b,c
R <- matrix(c(
  1, 0.8, 0.5, 0.3, 0.2,0.5,
  0.8, 1, 0.4, 0.3, 0.2,0.5,
  0.5, 0.4, 1, 0.3, 0.3,0.5,
  0.3, 0.3, 0.3, 1, 0.4,0.3,
  0.2, 0.2, 0.3, 0.4, 1,0.2,
  0.1, 0.8, 0.3, 0.4, 0.5, 1
), nrow = 6, ncol = 6)
a_tilde <- mvrnorm(n = 1, mu = rep(0, 6), Sigma = R)
a <- 0.2 * pnorm(a_tilde)
b_tilde <- mvrnorm(n = 1, mu = rep(0, 6), Sigma = R)
b <- 0.5 * pnorm(b_tilde)
c_tilde <- mvrnorm(n = 1, mu = rep(0, 6), Sigma = R)
c <- 0.1 * pnorm(c_tilde)

#Generate Train Data
gen_data(train_num, p=10,
     T.b0 = -3, T.beta = 0.5*c(1,-1,-2,-2, 1,-1,-2,-2, 1,-2), T.coef = 0.6,
     C.rate = 0.5,a,b,c, 
     filename =  train_file)

#Generate Test Data
gen_data(test_num, p=10, 
           T.b0 = -3, T.beta = 0.5*c(1,-1,-2,-2, 1,-1,-2,-2, 1,-2), T.coef = 0.6,
           C.rate = 0.5,a,b,c, 
           filename = test_file)

#Generate Embeddsing of W Variables
gen_emb(train_file,test_file,embeddings_file)

#Generate Fts Files
gen_fts(w_fts)
gen_fts(x_fts)
