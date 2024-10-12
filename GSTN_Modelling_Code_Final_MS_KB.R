
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

### selecting important vars

set.seed(51)
xgb_sel=xgboost(dtrain,
                max_depth=3,nrounds = 1000,objective = "binary:logistic",
                min_child_weight=1,verbose = F,eval_metric="auc",subsample=0.75,colsample_bytree=0.5,eta=0.01)
gc()

xgb_imp=xgb.importance(model=xgb_sel,feature_names = predictors) %>%data.frame()%>%mutate(cumul_gain=cumsum(Gain))
dim(xgb_imp) #22  5
xgb_vars= xgb_imp$Feature

(top_vars=xgb_imp %>% filter(cumul_gain<=.9995)%>% pull(Feature))
# [1] "Column18" "Column1"  "Column17" "Column7"  "Column4"  "Column5"  "Column14" "Column3"  "Column8"  "Column19" "Column6" 
# [12] "Column20" "Column2"  "Column15" "Column0"  "Column21" "Column12"

length(top_vars) #17

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
# 1 0.1642218   0.9678639   0.9976497

overall_data$pred_vals_final = ifelse(overall_data$pred_vals >0.1642218, 1, 0)

#### performance of train data

train_data = overall_data %>% filter(segment == "train") 

confusion_matrix = table(train_data$pred_vals_final,train_data$target)
confusion_matrix
#       0      1
# 0 688248    174
# 1  22852  73859

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


data.frame(accuracy=accuracy,
           precision=precision,
           sensitivity=sensitivity,
           specificity=specificity,
           auc=auc,
           f1_score=f1_score,
           logloss= logloss_val
)

# accuracy precision sensitivity specificity       auc  f1_score    logloss
# 1 0.9706725 0.7637084   0.9976497   0.9678639 0.9940543 0.9835472 0.05334938

#### performance of test data

test_data = overall_data %>% filter(segment == "test") 

confusion_matrix = table(test_data$pred_vals_final,test_data$target)
confusion_matrix
#       0      1
# 0 229307     50
# 1   7727  24628

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

data.frame(accuracy=accuracy,
           precision=precision,
           sensitivity=sensitivity,
           specificity=specificity,
           auc=auc,
           f1_score=f1_score,
           logloss= logloss_val
)

# accuracy precision sensitivity specificity       auc  f1_score    logloss
# 1 0.9702841 0.7611807   0.9979739   0.9674013 0.9940761 0.9833251 0.05335743

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

saveRDS(final_model_2,"GSTN_final_model_MS_KB.rds")

overall_data$pred_vals=predict(final_model_2,newdata = doverall) #dtrain


#### getting optimal cutoff
g1 <- roc(target ~ pred_vals, data = overall_data %>% filter(segment=="train"));g1
coords(g1, "best", transpose = FALSE)

### rounds=700
# threshold specificity sensitivity
# 1 0.1735756    0.973264   0.9965556

overall_data$pred_vals_final = ifelse(overall_data$pred_vals >0.1735756, 1, 0) 

#### performance of train data

train_data = overall_data %>% filter(segment == "train") 

confusion_matrix = table(train_data$pred_vals_final,train_data$target)
confusion_matrix
#      0      1
# 0 692088    255
# 1  19012  73778

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

data.frame(accuracy=accuracy,
           precision=precision,
           sensitivity=sensitivity,
           specificity=specificity,
           auc=auc,
           f1_score=f1_score,
           logloss= logloss_val
)

### rounds=700
# accuracy precision sensitivity specificity       auc  f1_score    logloss
# 1 0.9754602 0.7951072   0.9965556    0.973264 0.9960979 0.9862716 0.04344728

#### performance of test data

test_data = overall_data %>% filter(segment == "test") 

confusion_matrix = table(test_data$pred_vals_final,test_data$target)
confusion_matrix
#      0      1
# 0 230369    176
# 1   6665  24502

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

data.frame(accuracy=accuracy,
           precision=precision,
           sensitivity=sensitivity,
           specificity=specificity,
           auc=auc,
           f1_score=f1_score,
           logloss= logloss_val
)

### rounds=700
# accuracy precision sensitivity specificity       auc  f1_score    logloss
# 1 0.9738606  0.786152   0.9928681   0.9718817 0.9949141 0.9853693 0.04901414
