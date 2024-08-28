---
title: "MAP_simulation"
author: "Lechen Shen"
date: "2024-08-21"
output: html_document
---

```{r}
#STEP 1: loading 
#Install the package if you haven't

install.packages("MAP")

#Load the package

library(MAP)
```

```{r}
#STEP 2: Data preparation
# Generate 400 observations(patient records)
n = 400

# Generate ICD counts through a poisson regression, rpois(n/4,10) set the first n/4 patients ICD counts to be randomly selected from a Poisson regression(mean = 10);rpois(n/4,1) set the second n/4 patients ICD counts to be randomly selected from a Poisson regression(mean = 1);rpois(0,n/2) set all the left patients ICD counts to be 0.
ICD = c(rpois(n/4,10), rpois(n/4,1), rep(0,n/2) )

# Generate NLP counts through a poisson regression, rpois(n/4,10) set the first n/4 patients NLP counts to be randomly selected from a Poisson regression(mean = 10);rpois(n/4,1) set the second n/4 patients NLP counts to be randomly selected from a Poisson regression(mean = 1);rpois(0,n/2) set all the left patients NLP counts to be 0.
NLP = c(rpois(n/4,10), rpois(n/4,1), rep(0,n/2) )

#Combine the ICD counts and NLP counts into a sparse matrix
mat = Matrix(data=cbind(ICD,NLP),sparse = TRUE)

#Generate note counts through a Poisson regression, rpois(n,10)+5 set all the patients note counts to be randomly selected from a Poisson regression(mean = 10) and added by 5.
note = Matrix(rpois(n,10)+5,ncol=1,sparse = TRUE)

```

```{r}
#STEP 3: run the MAP algorithm
res = MAP(mat = mat,  note=note)

#The result is made up of two parts: scores(probability for each patient) and cut.MAP (threshold) 
head(res$scores)
res$cut.MAP
```