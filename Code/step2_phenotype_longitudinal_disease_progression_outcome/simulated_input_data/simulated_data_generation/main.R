# TODO: change the home dir
mdir = "C:/Users/NORTH/source/incident_phenotyping/"

library(MAP)
library(ff)
library(data.table)
library(text2vec)

args = commandArgs(trailingOnly = TRUE)


total_data_num = as.numeric(args[1])
# 37736
labeled_data_num = as.numeric(args[2])
# 1000
num.train = args[3]

#"cum": cumulative counts; "stacked": stacked counts
num.test = args[4]

#----------------------------------------
# data generation #
#----------------------------------------
source(paste0(mdir,'/simulation_data_functions_v4.R'))
source(paste0(mdir,"/get_Folds.R"))

A<-matrix(rep(c(1,-2),50),10)
pmean <- function(x,y) (x+y)/2
A[] <- pmean(A, matrix(A, nrow(A), byrow=TRUE))
s<-matrix(rep(1,100),10)-2*A+diag(rep(c(0.5,-4.5),5))


Sigma_filename = 'data/Sigma.csv'
Sigma = read.csv(paste0(mdir, Sigma_filename))
Norms_filename = 'data/Norms.csv'
Norms = read.csv(paste0(mdir, Norms_filename))

# print(Sigma)


#Generate Data


gen_data_2(total_data_num, p=10, Sigma, Norms, T.family = 'linear',
     T.b0 = -3, T.beta = 0.5*c(1,-1,-2,-2, 1,-1,-2,-2, 1,-2), T.coef = 0.05,
     C.rate = 0.5, filename = paste0(mdir,'Simulation/SimDat/SimDat.1/','SimDat.1.csv'),dataname =paste0(mdir,'Simulation/SimDat/SimDat.1/','SimDat.1.Rds'))


