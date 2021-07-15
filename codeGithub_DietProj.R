# load packages ####
require(ChemometricsWithR); require(glmnet); require(MASS); require(caret); require(gam)
require(pls); require(tree); require(randomForest); require(ada)
require(HDclassif); require(ipred); require(rpart); require(mclust)
require(doMC); require(seriation); require(groupdata2); require(chemometrics)

# some useful functions ####
cv.folds <- function(n, nfolds = 5) {
  # Create 'nfolds' groups of approximately the same size. 
  # Returns a list with 'nfolds' elements containing indexes of each fold.
  n <- as.integer(n)
  nfolds <- as.integer(min(n, nfolds))
  folds <- split(sample(n, size = n, replace = FALSE),
                 rep(seq_len(nfolds), length.out = n))
  return(folds)
}  

postproc <- function(pred, true) {
  t <- table(pred, true)
  err <- 1-(sum(diag(t))/length(true))
  accuracy <- 1-err
  sensitivity <- t[1]/(t[1] + t[2])
  specificity <- t[4]/(t[3] + t[4])
  F1 <- (2*t[1])/(2*t[1] + t[2] + t[3])
  mat <- matrix(c(accuracy, sensitivity, specificity, F1), ncol = 4)
  colnames(mat) <- c("accuracy","sensitivity","specificity", "F1-score")
  return(mat) 
}



# MAY-AUGUST ANALYSES ####

# load data #### 
dati <- readxl::read_xlsx("MIR Pasture vs TMR 2015 - 2017 May June July August only.xlsx")


# build dataset ####
spectra <- as.matrix(dati[,21:dim(dati)[2]],ncol=1060)

# work in log10 scale, everything afterwards being done with this scale
spectra <- t(apply(spectra,1,function(x) log10(1/x)))

# outlier detection 

# check visually if something wrong is going on 
plot(1:1060, spectra[1,], type = "n", ylim = c(min(spectra), max(spectra)))
for (i in 1:1060) lines(spectra[i,], lwd = 0.7, lty = 2)

pca <- princomp(spectra)
perc_var <- cumsum(pca$sdev^2/sum(pca$sdev^2))
# 8 principal components to explain the 90% of the variance 
pc_scores <- pca$scores[,1:8]
detect <- Moutlier(pc_scores, quantile = 0.99, plot = F)
sub_out <- which(detect$md >= detect$cutoff)

# remove outlier
spectra_red <- spectra[-sub_out,]
# check visually --> ok
plot(1:1060, spectra_red[1,], type = "n", ylim = c(min(spectra_red), max(spectra_red)))
for (i in 1:1060) lines(spectra_red[i,], lwd = 0.7, lty = 2)

# build variable to predict
diet <- dati$`Feed type`
diet <- as.factor(diet)
diet <- diet[-sub_out]
cowid <- dati$Cow
cowid <- cowid[-sub_out]

# remove water wavelengths 
water_wave <- c(174:207,538:720,751:1060)
spectra <- spectra_red[,-water_wave]



# Split in training and validation set ####
set.seed(2906)
df <- as.data.frame(spectra)
df$id_cow <- as.factor(cowid)
df$diet <- diet
split <- groupdata2::partition(data = df, p = 0.6,id_col = "id_cow")

train <- split[[1]]
test <- split[[2]]
Ytrain <- train$diet
Ytest <- test$diet
Ytrain_num <- ifelse(Ytrain=="Pasture", 1, 0)
Ytest_num <- ifelse(Ytest=="Pasture", 1, 0)
Xtrain <- train[-1]; Xtrain$diet <- NULL
train_X <- as.matrix(Xtrain, nrow = length(Ytrain), ncol = ncol(spectra))
Xtest <- test[-1]; Xtest$diet <- NULL
test_X <- as.matrix(Xtest, nrow = length(Ytest), ncol = ncol(spectra))

df_train <- data.frame(Y = Ytrain, X = train_X)
df_test <- data.frame(Y = Ytest, X = test_X)



# CLASSIFIERS ####

# M1 --- LASSO penalized logistic regression ####
set.seed(1991)
m2 <- cv.glmnet(train_X, Ytrain, alpha=1, family="binomial", type.measure = "class", parallel = T)
p2 <- predict(m2, s="lambda.min",newx = test_X,type="class")  
(res2 <- postproc(p2, Ytest)) 

p2train <- predict(m2, s = "lambda.min", newx = train_X, type = "class")
(res2train <- postproc(p2train, Ytrain)) 

coef_m2 <- coef(m2,s = "lambda.min") 
length(coef_m2[which(coef_m2!=0)])-1

# M2 --- RIDGE penalized logistic regression ####
set.seed(1992) 
m3 <- cv.glmnet(train_X, Ytrain, alpha=0, family="binomial", type.measure = "class", parallel = T)
p3 <- predict(m3, s="lambda.min",newx = test_X,type="class")  
(res3 <- postproc(p3, Ytest)) 

p3train <- predict(m3, s = "lambda.min", newx = train_X, type = "class")
(res3train <- postproc(p3train, Ytrain)) 

# M3 --- ELASTIC-NET penalized logistic regression ####
set.seed(1993)
cv_fold = trainControl(method = "cv", number = 4)
def_elnet = caret::train(
  Y ~ ., data = df_train,
  method = "glmnet",
  trControl = cv_fold)
m4 <- glmnet(train_X, Ytrain, alpha = 0.1, family = "binomial", lambda = 0.0001779845)
p4 <- predict(m4, newx = test_X, type = "class")
(res4 <- postproc(p4, Ytest))

p4train <- predict(m4, newx = train_X, type = "class")
(res4train <- postproc(p4train, Ytrain)) 

# M4 --- LINEAR DISCRIMINANT ANALYSIS ####
m5 <- MASS::lda(Y ~ ., data = df_train)
p5 <- predict(m5,newdata = df_test)
(res5 <- postproc(p5$class, Ytest))

p5train <- predict(m5)
(res5train <- postproc(p5train$class, Ytrain))

# M5 --- MODEL-BASED DISCRIMINANT ANALYSIS ####
m7 <- MclustDA(train_X, Ytrain, modelType = "EDDA")
p7 <- predict(m7, newdata = test_X)
(res7 <- postproc(p7$classification, Ytest)) 

p7train <- predict(m7)
(res7train <- postproc(p7train$classification, Ytrain)) 

# M6 --- PLSDA ####
ctrl <- trainControl(method = "repeatedcv", number = 4, repeats = 1, verboseIter = TRUE, 
                     classProbs = TRUE, summaryFunction = twoClassSummary, allowParallel = TRUE)
m9 <- caret::train(Y ~ ., df_train,  method = "pls", preProc =c("scale", "center"),
                   tuneLength = 200, trControl = ctrl)
p9 <- predict(m9, newdata = df_test)
(res9 <- postproc(p9, Ytest))

p9train <- predict(m9)
(res9train <- postproc(p9train, Ytrain))

# M7 --- PCR ####
pca <- princomp(train_X, cor = T)
perc_var_train <- cumsum(pca$sdev^2/sum(pca$sdev^2))
pcatrain <- pca$scores[,1:4] 
#explaining more than 95% of the variability 
pcatest <- predict(pca, test_X)
pcatest <- pcatest[,1:4]
df_pca_train <- data.frame(Y=Ytrain, X=pcatrain)
df_pca_test <- data.frame(Y=Ytest, X=pcatest)

m10 <- glm(Y~., data = df_pca_train, family = "binomial")
p10 <- predict(m10, newdata = df_pca_test, type = "response")
p10 <- p10 > .5
(res10 <- postproc(p10, Ytest))


# M8 --- RANDOM FOREST ###
set.seed(061991)
m11 <- randomForest(Y ~ ., data = df_train, importance = T, ntree = 1000)
p11 <- predict(m11, df_test, type = "class")
(res11 <- postproc(p11, Ytest))

p11train <- predict(m11, type = "class")
(res11train <- postproc(p11train, Ytrain))

# M9 --- BOOSTING ### 
m12 <- ada::ada(Y ~., data = df_train, iter=500, nu = 1)
p12 <- predict(m12, df_test)
(res12 <- postproc(p12, Ytest))

p12train <- predict(m12, df_train)
(res12train <- postproc(p12train, Ytrain))


# M10 --- SUPPORT VECTOR MACHINES #### 
require(e1071) 
tune.svm <- tune(svm, Y ~ ., data = df_train, kernel = "radial",
                 ranges = list(gamma = seq(from = 0.00005, to = 0.0005, length.out = 8), 
                               cost = seq(from = 4000, to= 10000, length.out = 8)))
m13 <- svm(Y ~ ., data = df_train, kernel = "radial", gamma = tune.svm$best.parameters$gamma, 
           cost = tune.svm$best.parameters$cost)
p13 <- predict(m13, df_test)
(res13 <- postproc(p13, Ytest))

