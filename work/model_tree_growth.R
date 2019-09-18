## Aim: Develop a model that is able to predict tree growth across Europe
rm(list=ls())
library(randomForest)
library(caret)
library(CAST)
library(raster)
library(sptm)
library(ggplot2)
library(parallel)
library(doParallel)
library(gbm)
library(viridis)
library(ggplot2)
################################################################################
# Settings
################################################################################
cores <- 7
response <- "BAI"
tuneLength <- 3
subset <- 0.3 ### CHANGE SUBSET 
mainpath <-"/home/hanna/Documents/Projects/ML4GenTree/treeGrowthModel/data/"

#mainpath <- "/scratch/tmp/hmeyer1/ML4GenTree/"
#resultpath <- mainpath

################################################################################
# Read data
################################################################################

dat <- read.csv(paste0(mainpath,"fagus_MSlearning4.csv"))
################################################################################
# Subset (to speed things up?)
################################################################################

set.seed(100)
trainids <- createDataPartition(dat$sample,list=FALSE,p=subset)
trainDat <- dat[trainids,]


predictors <- names(dat)[c(10:146,206)] # CHANGE VARIABLES HERE!!!!

trainDat <- trainDat[complete.cases(trainDat[,c(response,predictors)]),]

################################################################################
# Prepare cross-validation (leave-population-out)
################################################################################
set.seed(100)
folds <- CreateSpacetimeFolds(trainDat, spacevar="population", 
                              k=length(unique(trainDat$population)))

ctrl <- trainControl(method="cv",
                     savePredictions = TRUE,
                     index=folds$index,
                     indexOut=folds$indexOut)
# non spatial CV
ctrl_nsp <- trainControl(method="cv",
                     savePredictions = TRUE)
################################################################################
# First Model training
################################################################################

cl <- makeCluster(cores)
registerDoParallel(cl)

set.seed(100)

############# RANGER VERSION
#model <- train(trainDat[,predictors],
#               trainDat[,response],
#               method="ranger",  #ranger-rf instead of rf implementation beacause faster!
#               metric="RMSE", #we might need to change that because WITHIN Side
#               trControl=ctrl,
#               tuneLength=tuneLength,
#               importance = 'impurity')



############# GBM VERSION
model <- train(trainDat[,predictors],
               trainDat[,response],
               method="gbm",  #ranger-rf instead of rf implementation beacause faster!
               metric="Rsquared", #we might need to change that because WITHIN Side
               trControl=ctrl,
               tuneLength=tuneLength)


stopCluster(cl)
save(model,file=paste0(resultpath,"/model_",response,".RData"))


################################################################################
# Model validation
################################################################################
## needs adaptation depending on algorithm (here for gbm)
cvPredictions <- model$pred[c(model$pred$n.trees==model$bestTune$n.trees&
                              model$pred$interaction.depth==model$bestTune$interaction.depth,
                            model$pred$shrinkage==model$bestTune$shrinkage&
                              model$pred$n.minobsinnode==model$bestTune$n.minobsinnode),
                            c("obs","pred")]
 regressionStats(cvPredictions$obs,cvPredictions$pred)
# 
# ################################################################################
# # Visualisation
# ################################################################################
# 
print(ggplot(cvPredictions, aes(obs,pred)) +
        stat_binhex(bins=100)+
        xlim(min(cvPredictions),max(cvPredictions))+
        ylim(min(cvPredictions),max(cvPredictions))+
        xlab("Measured Tree Growth")+
        ylab("Predicted Tree Growth")+
        geom_abline(slope=1, intercept=0,lty=2)+
        scale_fill_gradientn(name = "data points",
                             #trans = "log",
                             #breaks = 10^(0:3),
                             colors=viridis(10)))
