
########################################### GSTN ML Model Development  ########################################### 

#### Importing required libraries
library(dplyr)
library(ROCR)
library(pROC)
library(xgboost)
library(purrr)
library(MLmetrics)
library(yardstick)

#### Setting working directory
setwd("~/Documents/GSTN Hackathon")

#### custom functions
LogLoss=function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

#### Reading train & test data
X_Train_Data_Input <- read.csv("~/Documents/GSTN Hackathon/X_Train_Data_Input.csv")
Y_Train_Data_Target <- read.csv("~/Documents/GSTN Hackathon/Y_Train_Data_Target.csv")

X_Test_Data_Input <- read.csv("~/Documents/GSTN Hackathon/Test_20/Test_20/X_Test_Data_Input.csv")
Y_Test_Data_Target <- read.csv("~/Documents/GSTN Hackathon/Test_20/Test_20/Y_Test_Data_Target.csv")

#### checking dim
dim(X_Train_Data_Input) #785133     23
dim(Y_Train_Data_Target) #785133      2

dim(X_Test_Data_Input) #261712     23
dim(Y_Test_Data_Target) #261712      2

#### Creating overall data
train_overall_data= X_Train_Data_Input %>%left_join(Y_Train_Data_Target) %>% mutate(segment="train") #by = join_by(ID)
dim(train_overall_data) #785133     25

test_overall_data = X_Test_Data_Input %>%left_join(Y_Test_Data_Target) %>% mutate(segment="test")
dim(test_overall_data) #261712     25

overall_data =  rbind(train_overall_data,test_overall_data) 
dim(overall_data) # 1046845      25

#### Checking bad rates

overall_data %>% group_by(segment) %>% summarise(total=n(),bads=sum(target,na.rm = T))%>% mutate(bad_rate=round(100*bads/total,2))
# segment  total  bads bad_rate
# 1 test    261712 24678     9.43
# 2 train   785133 74033     9.43

#### Checking NA count & NA perc
na_cnt = data.frame(var_name=names(colSums(is.na(overall_data))),cnt=colSums(is.na(overall_data)))
rownames(na_cnt)=NULL
na_cnt%>%filter(cnt>0)%>%mutate(perc=round(100*cnt/nrow(overall_data),2))%>%arrange(desc(perc))
#   var_name    cnt  perc
# 1  Column9 975990 93.23
# 2 Column14 487382 46.56
# 3  Column5 222839 21.29
# 4  Column4 170420 16.28
# 5  Column3 168537 16.10
# 6 Column15  21941  2.10
# 7  Column6   5084  0.49
# 8  Column8   5084  0.49
# 9  Column0     11  0.00

########## imputing NA values by average in train data
overall_data$Column0[is.na(overall_data$Column0)==TRUE]=mean(train_overall_data$Column0[is.na(train_overall_data$Column0)==FALSE]) # 0
overall_data$Column3[is.na(overall_data$Column3)==TRUE]=mean(train_overall_data$Column3[is.na(train_overall_data$Column3)==FALSE]) # 0.6781394
overall_data$Column4[is.na(overall_data$Column4)==TRUE]=mean(train_overall_data$Column4[is.na(train_overall_data$Column4)==FALSE]) # 0.7014035
overall_data$Column5[is.na(overall_data$Column5)==TRUE]=mean(train_overall_data$Column5[is.na(train_overall_data$Column5)==FALSE]) # -0.00746865
overall_data$Column6[is.na(overall_data$Column6)==TRUE]=mean(train_overall_data$Column6[is.na(train_overall_data$Column6)==FALSE]) # -0.4079391
overall_data$Column8[is.na(overall_data$Column8)==TRUE]=mean(train_overall_data$Column8[is.na(train_overall_data$Column8)==FALSE]) # 0.1220851
overall_data$Column9[is.na(overall_data$Column9)==TRUE]=mean(train_overall_data$Column9[is.na(train_overall_data$Column9)==FALSE]) # -0.08182017
overall_data$Column14[is.na(overall_data$Column14)==TRUE]=mean(train_overall_data$Column14[is.na(train_overall_data$Column14)==FALSE]) # 0.001350606
overall_data$Column15[is.na(overall_data$Column15)==TRUE]=mean(train_overall_data$Column15[is.na(train_overall_data$Column15)==FALSE]) #0.003390099


########################## Modelling started - Using All vars ########################## 

predictors=overall_data %>% select(Column0:Column21)%>%colnames()
length(predictors) #22

zeroVariance= overall_data %>%select(one_of(predictors))%>%summarise_all(var)%>%select_if(function(.).==0) %>%colnames()
zeroVariance

predictors=setdiff(predictors,zeroVariance)
length(predictors) #22

dtrain = xgb.DMatrix(data.matrix(overall_data[overall_data$segment=="train",predictors]),
                     label=overall_data[overall_data$segment=="train",][["target"]])

doverall = xgb.DMatrix(data.matrix(overall_data[,predictors]),
                       label=overall_data[["target"]])

### selecting important vars / feature selection

set.seed(51)
xgb_sel=xgboost(dtrain,
                max_depth=3,nrounds = 700,objective = "binary:logistic",
                min_child_weight=1,verbose = F,eval_metric="auc",subsample=0.75,colsample_bytree=0.5,eta=0.01)
gc()

xgb_imp=xgb.importance(model=xgb_sel,feature_names = predictors) %>%data.frame()%>%mutate(cumul_gain=cumsum(Gain))
dim(xgb_imp) #22  5
xgb_vars= xgb_imp$Feature

xgb_imp
# Feature         Gain        Cover    Frequency cumul_gain
# 1  Column18 7.532772e-01 0.2034584426 0.1125478927  0.7532772
# 2   Column1 1.307029e-01 0.2723176584 0.2772988506  0.8839801
# 3  Column17 2.755278e-02 0.0631700888 0.0625000000  0.9115329
# 4   Column4 1.962328e-02 0.0884798024 0.0730363985  0.9311562
# 5   Column7 1.937601e-02 0.0373394192 0.0804597701  0.9505322
# 6  Column14 1.036323e-02 0.0871759568 0.0742337165  0.9608954
# 7  Column19 8.373582e-03 0.0344074632 0.0217911877  0.9692690
# 8   Column8 7.172288e-03 0.0600432601 0.0490900383  0.9764413
# 9   Column5 6.888451e-03 0.0237882628 0.0383141762  0.9833297
# 10  Column3 6.244543e-03 0.0465915611 0.0727969349  0.9895743
# 11  Column6 4.012592e-03 0.0193871892 0.0457375479  0.9935869
# 12 Column20 2.704474e-03 0.0226898286 0.0148467433  0.9962913
# 13 Column15 1.167231e-03 0.0122846766 0.0225095785  0.9974586
# 14 Column21 5.514258e-04 0.0078416394 0.0045498084  0.9980100
# 15 Column12 4.616169e-04 0.0058171552 0.0076628352  0.9984716
# 16  Column2 4.345735e-04 0.0046343395 0.0146072797  0.9989062
# 17  Column0 3.606562e-04 0.0030695962 0.0076628352  0.9992668
# 18 Column11 2.608564e-04 0.0028071331 0.0045498084  0.9995277
# 19 Column10 1.850266e-04 0.0012755374 0.0040708812  0.9997127
# 20 Column13 1.713449e-04 0.0022744840 0.0038314176  0.9998841
# 21  Column9 1.061289e-04 0.0007745316 0.0069444444  0.9999902
# 22 Column16 9.805767e-06 0.0003719739 0.0009578544  1.0000000

(top_vars=xgb_imp %>% filter(cumul_gain<=.9999)%>% pull(Feature))
# [1] "Column18" "Column1"  "Column17" "Column4"  "Column7"  "Column14" "Column19" "Column8"  "Column5"  "Column3"  "Column6" 
# [12] "Column20" "Column15" "Column21" "Column12" "Column2"  "Column0"  "Column11" "Column10" "Column13"

length(top_vars) #20

predictors= top_vars

dtrain = xgb.DMatrix(data.matrix(overall_data[overall_data$segment=="train",predictors]),
                     label=overall_data[overall_data$segment=="train",][["target"]])

doverall = xgb.DMatrix(data.matrix(overall_data[,predictors]),
                       label=overall_data[["target"]])

########### First iteration 

params_1 = list(objective = "binary:logistic",
                eval_metric="auc",
                max_depth=3,
                eta=0.01,
                gamma=5,
                colsample_bytree=0.5,
                min_child_weight=1,
                subsample=0.8
                # ,lambada=0.1
)

xgb_model=xgb.train(params = params_1,data = dtrain,nrounds = 1000,
                    #early_stopping_rounds = 50,
                    print_every_n = 10,verbose = 1L)

class(xgb_model)
# saveRDS(xgb_model,"xgb_model_V1.rds")

xgb_model= readRDS("xgb_model_V1.rds")

overall_data$pred_vals=predict(xgb_model,newdata = doverall) #dtrain

#### getting optimal cutoff
g1 <- roc(target ~ pred_vals, data = overall_data %>% filter(segment=="train"));g1
coords(g1, "best", transpose = FALSE)
# threshold specificity sensitivity
# 1 0.1399288   0.9667318   0.9984872

overall_data$pred_vals_final = ifelse(overall_data$pred_vals >0.1399288, 1, 0)

#### performance of train data

train_data = overall_data %>% filter(segment == "train") 

confusion_matrix = table(train_data$pred_vals_final,train_data$target)
confusion_matrix
#       0      1
# 0 687443    112
# 1  23657  73921

accuracy  = (confusion_matrix[1,1]+confusion_matrix[2,2])/nrow(train_data)
precision = (confusion_matrix[2,2])/sum(train_data$pred_vals_final)
sensitivity = (confusion_matrix[2,2])/sum(train_data$target)
specificity = (confusion_matrix[1,1])/(nrow(train_data)-sum(train_data$target))

# ROC-AUC Curve
ROCPred <- prediction(train_data$pred_vals, train_data$target)
ROCPer <- performance(ROCPred, measure = "tpr", x.measure = "fpr")
auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
f1_score=F1_Score(train_data$target,train_data$pred_vals_final)
logloss_val=LogLoss(train_data$target,train_data$pred_vals)
#bal_accuracy(truth = as.numeric(train_data$target),estimate = (train_data$pred_vals_final))


train_performance=data.frame(accuracy=accuracy,
                             precision=precision,
                             sensitivity=sensitivity,
                             specificity=specificity,
                             auc=auc,
                             f1_score=f1_score,
                             logloss= logloss_val
)
train_performance
# accuracy precision sensitivity specificity       auc  f1_score    logloss
# 1 0.9697261 0.7575581   0.9984872   0.9667318 0.9940086 0.9830058 0.05346147

#### performance of test data

test_data = overall_data %>% filter(segment == "test") 

confusion_matrix = table(test_data$pred_vals_final,test_data$target)
confusion_matrix
#       0      1
# 0 229051     36
# 1   7983  24642

accuracy  = (confusion_matrix[1,1]+confusion_matrix[2,2])/nrow(test_data)
precision = (confusion_matrix[2,2])/sum(test_data$pred_vals_final)
sensitivity = (confusion_matrix[2,2])/sum(test_data$target)
specificity = (confusion_matrix[1,1])/(nrow(test_data)-sum(test_data$target))

# ROC-AUC Curve
ROCPred <- prediction(test_data$pred_vals, test_data$target)
ROCPer <- performance(ROCPred, measure = "tpr", x.measure = "fpr")
auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
f1_score=F1_Score(test_data$target,test_data$pred_vals_final)
logloss_val=LogLoss(test_data$target,test_data$pred_vals)
#bal_accuracy(truth = as.numeric(test_data$target),estimate = (test_data$pred_vals_final))

Test_performance=data.frame(accuracy=accuracy,
           precision=precision,
           sensitivity=sensitivity,
           specificity=specificity,
           auc=auc,
           f1_score=f1_score,
           logloss= logloss_val
)
Test_performance
# accuracy precision sensitivity specificity       auc  f1_score    logloss
# 1 0.9693594 0.7553103   0.9985412   0.9663213 0.9940496 0.9827963 0.05339355

################################# Hyper parameter tuning ################################# 

# Set initial parameters for binary classification
params <- list(
  objective = "binary:logistic",    # For binary classification
  eval_metric = "logloss"           # Log-loss for binary classification
)

# Perform cross-validation
cv_results <- xgb.cv(
  params = params,
  data = dtrain,
  nfold = 5,                        # 5-fold cross-validation
  nrounds = 100,                    # Up to 100 boosting rounds
  early_stopping_rounds = 10,       # Stop if no improvement after 10 rounds
  verbose = TRUE                    # Print results to console
)

# Extract the Best Number of Rounds
best_nrounds <- cv_results$best_iteration

# Train the Final Model with Optimal Parameters
final_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds #83
)

# Grid search for hyperparameter tuning
search_grid <- expand.grid(
  max_depth = c(3, 6),
  eta = c(0.01, 0.1),
  colsample_bytree = c(0.5, 0.7)
)

best_log_loss <- Inf  # We want to minimize log-loss, so start with a high value
best_params <- list()

for (i in 1:nrow(search_grid)) {
  params <- list(
    objective = "binary:logistic",  # For binary classification
    eval_metric = "logloss",        # Use log-loss for evaluation
    max_depth = search_grid$max_depth[i],
    eta = search_grid$eta[i],
    colsample_bytree = search_grid$colsample_bytree[i]
  )
  
  # Cross-validation for current set of parameters
  cv_results <- xgb.cv(
    params = params,
    data = dtrain,
    nfold = 5,                      # 5-fold cross-validation
    nrounds = 100,                  # Up to 100 boosting rounds
    early_stopping_rounds = 10,     # Stop early if no improvement
    verbose = TRUE
  )
  
  # Extract the minimum log-loss
  min_logloss <- min(cv_results$evaluation_log$test_logloss_mean)
  
  # Update best parameters if current log-loss is lower
  if (min_logloss < best_log_loss) {
    best_log_loss <- min_logloss
    best_params <- params
    best_nrounds <- cv_results$best_iteration
  }
  print(i)
  print(best_params)
  print(best_nrounds)
  
}

saveRDS(best_params,"best_params.rds")
best_nrounds =100

print(best_params)
# $objective
# [1] "binary:logistic"
# 
# $eval_metric
# [1] "logloss"
# 
# $max_depth
# [1] 6
# 
# $eta
# [1] 0.1
# 
# $colsample_bytree
# [1] 0.7


final_model_2 <- xgb.train(
  objective="binary:logistic",
  eval_metric="logloss",
  max_depth=6,
  eta=0.1,
  colsample_bytree=0.7,
  data = dtrain,
  nrounds = 700
)

# saveRDS(final_model_2,"GSTN_final_model_MS_KB.rds")

overall_data$pred_vals=predict(final_model_2,newdata = doverall) #dtrain


#### getting optimal cutoff
g1 <- roc(target ~ pred_vals, data = overall_data %>% filter(segment=="train"));g1
coords(g1, "best", transpose = FALSE)

# threshold specificity sensitivity
# 1 0.1811275    0.973908   0.9973795

### rounds=700
# threshold specificity sensitivity
# 1 0.1735756    0.973264   0.9965556

overall_data$pred_vals_final = ifelse(overall_data$pred_vals >0.1811275, 1, 0) 

#### performance of train data

train_data = overall_data %>% filter(segment == "train") 

confusion_matrix = table(train_data$pred_vals_final,train_data$target)
confusion_matrix
# 0      1
# 0 692546    194
# 1  18554  73839

accuracy  = (confusion_matrix[1,1]+confusion_matrix[2,2])/nrow(train_data)
precision = (confusion_matrix[2,2])/sum(train_data$pred_vals_final)
sensitivity = (confusion_matrix[2,2])/sum(train_data$target)
specificity = (confusion_matrix[1,1])/(nrow(train_data)-sum(train_data$target))

# ROC-AUC Curve
ROCPred <- prediction(train_data$pred_vals, train_data$target)
ROCPer <- performance(ROCPred, measure = "tpr", x.measure = "fpr")
auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
f1_score=F1_Score(train_data$target,train_data$pred_vals_final)
logloss_val=LogLoss(train_data$target,train_data$pred_vals)
#bal_accuracy(truth = as.numeric(train_data$target),estimate = (train_data$pred_vals_final))

train_performance=data.frame(accuracy=accuracy,
                             precision=precision,
                             sensitivity=sensitivity,
                             specificity=specificity,
                             auc=auc,
                             f1_score=f1_score,
                             logloss= logloss_val
)
train_performance
# accuracy precision sensitivity specificity       auc  f1_score    logloss
# 1 0.9761212 0.7991839   0.9973795    0.973908 0.9964086 0.9866452 0.04192136

#### performance of test data

test_data = overall_data %>% filter(segment == "test") 

confusion_matrix = table(test_data$pred_vals_final,test_data$target)
confusion_matrix
# 0      1
# 0 230484    196
# 1   6550  24482

accuracy  = (confusion_matrix[1,1]+confusion_matrix[2,2])/nrow(test_data)
precision = (confusion_matrix[2,2])/sum(test_data$pred_vals_final)
sensitivity = (confusion_matrix[2,2])/sum(test_data$target)
specificity = (confusion_matrix[1,1])/(nrow(test_data)-sum(test_data$target))

# ROC-AUC Curve
ROCPred <- prediction(test_data$pred_vals, test_data$target)
ROCPer <- performance(ROCPred, measure = "tpr", x.measure = "fpr")
auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
f1_score=F1_Score(test_data$target,test_data$pred_vals_final)
logloss_val=LogLoss(test_data$target,test_data$pred_vals)

test_performance=data.frame(accuracy=accuracy,
                            precision=precision,
                            sensitivity=sensitivity,
                            specificity=specificity,
                            auc=auc,
                            f1_score=f1_score,
                            logloss= logloss_val
)
test_performance
# accuracy precision sensitivity specificity       auc  f1_score    logloss
# 1 0.9742236 0.7889276   0.9920577   0.9723668 0.9949046 0.9855767 0.04898082

