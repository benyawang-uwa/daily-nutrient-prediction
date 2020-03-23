""" a hybrid machine learning model for the prediction of daily surface water nutrient 
    concentration 

# description: this script is used to generate daily nutrient concentration

# date: 23/03/2020
# name: Benya Wang

"""

## load R packages  
library(mlr)
library(EcoHydRology)
library(parallelMap)
library(caret)
library(magrittr)
library(xts)
library(reshape2)
library(h2o)
library(lubridate)
library(zoo) 
library(dplyr)
library(JBTools)
library(EGRET)

# start the h2o cluster  
h2o.init(nthreads = 8,max_mem_size = "20g")

# define the nutrient list 
all_n_target=c("TP","TN","DOC","DON","DIN","FRP")

# function to combine day, month, year in the dataset and convert as date 
create_date<-function(data_set){
  data_set[,"Date"]=paste0(data_set[,"Day"],"/",data_set[,"Month"],"/",data_set[,"Year"])
  data_set$Date<-dmy(data_set$Date)
  data_set[,1:3]<-NULL
  return(data_set)
}

# function to scale the data 
standardize<-function(train_data,test_data){
  scaled_train=train_data
  scaled_test=test_data   
  for(ii in seq(1,ncol(scaled_train),1)){    
    min_v<-min(train_data[,ii])    
    max_v<-max(train_data[,ii])    
    scaled_train[,ii]=(train_data[,ii]-min_v)/(max_v-min_v)
    scaled_test[,ii]=(test_data[,ii]-min_v)/(max_v-min_v)
  }
  return(list(scaled_train,scaled_test))
}

# function to reverse scaling 
standardize_reverse<-function(data_set,max_v,min_v){
  data_set<-data_set*(max_v-min_v)+min_v
  return(data_set)
}


# function to merge nutrient, flow, and rainfall datasets 
combined_nutrient_unscaled <- function(dataset, Flow_DC, Rainfall, n_target) {
  
  Nutrient_target <- dataset[is.na(dataset[[n_target]]) == F, c("Date", n_target)]
  Nutrient_target[["Date"]] <- ymd(Nutrient_target[["Date"]])
  
  names(Nutrient_target) <- c("Date", n_target)
  
  # convert date
  Nutrient_target$Date<-as.character(Nutrient_target$Date)
  Flow_DC$Date<-as.character(Flow_DC$Date)
  Rainfall$Date<-as.character(Rainfall$Date)
  
  # merge datasets 
  combined_target <- merge(Nutrient_target, Flow_DC, by = 'Date', all.y = T) 
  combined_target<-merge(Rainfall,combined_target, by = 'Date', all.y = T)  
  
  combined_target$Date<-ymd(combined_target$Date)

  ## add the month, the day of the year, and the week of the year to the combined dataset 
  combined_target$month <- as.numeric(lubridate::month(combined_target$Date))
  combined_target$yday <- as.numeric(yday(combined_target$Date))
  combined_target$week <- as.numeric(lubridate::week(combined_target$Date))
  
  combined_target$Date <- as.numeric(combined_target$Date)
  
  # Normalize the nutrient_target data  
  combined_target<-combined_target[order(combined_target$Date),]
  
  # three passes baseflow separation 
  baseflow_sep<-data.frame(Date=as.Date(combined_target$Date, origin="1970-01-01"),
                           P_mm=combined_target$Rainfall,
                           Streamflow_m3s=combined_target$DC_mean)
  
  bfs<-BaseflowSeparation(baseflow_sep$Streamflow_m3s, passes=3)
  
  # generate baseflow and quick flow features
  for (d in c(2, 3)) {
    bfs[, paste0("mean_QF", as.character(d))] <- rollmeanr(bfs[,2], k = d, fill = 0)
  }  
  
  for (d in c(2, 3)) {
    bfs[, paste0("mean_BF", as.character(d))] <- rollmeanr(bfs[,1], k = d, fill = 0)
  }
  combined_target<-cbind(combined_target,bfs)
  
  # generate some temporal features 
  combined_target$DecYear<-decimal_date(as.Date(combined_target$Date, origin="1970-01-01"))
  combined_target$SinDY<--sin(2*pi*combined_target$DecYear) 
  combined_target$CosDY<-cos(2*pi*combined_target$DecYear)
  combined_target$logQ<-log(combined_target$DC_mean+0.001)
  
  ## Qbaseflow/Qtotal
  combined_target$Qbf_Qtotal<-combined_target$bt/(combined_target$DC_mean+0.001)
  combined_target$Qqf_Qtotal<-combined_target$qft/(combined_target$DC_mean+0.001)
  
  combined_target$DecYear<-NULL
  
  return(combined_target)
  
}

# function to build h2o gbm model.
model_build <- function(dataset, n_target) {
  
  y <- n_target
  x <- setdiff(names(dataset), y)
  
  dataset<-as.h2o(dataset)
  # just change to h2o.randomForest for randomForest model 
  gbm <- h2o.gbm(x = x,
                 y = y,
                 training_frame = dataset,
                 nfolds = 5,
                 seed = 719)
  return(gbm)
}

# build a simple model to select important 
find_importVar<-function(train_data, test_data, n_target, var_number){
  
  rf_var<- model_build(as.h2o(train_data), n_target)
  var_imp<-h2o.varimp(rf_var)
  
  import_var<-var_imp[1:var_number,1]
  
  if (!("Date" %in% import_var)) {
    import_var<-c(import_var,n_target,"Date")
  } else {
    import_var<-c(import_var,n_target)
  }
  
  training_out<-train_data[,import_var]
  testing_out<-test_data[,import_var]
  
  high_cor<-findCorrelation(cor(training_out), cutoff = 0.9, verbose = F)
  
  if (length(high_cor)==0){
    training_out <- training_out
    testing_out <- testing_out
    
  } else{
    training_out <- training_out[,-high_cor]
    testing_out <- testing_out[,-high_cor]
  }
  
  if (!("Date" %in% names(training_out))) {
    training_out$Date<-train_data$Date
    testing_out$Date<-test_data$Date
  } 
  
  out_data<-list(training_out,testing_out)
  
  return(out_data)
}

# function to build a linear model 
lm_model<-function(train_data,test_data,n_target){
  
  ## method 1: linear model  
  y <- train_data[,n_target]
  x <- train_data[,"DC_mean"]
  model_lm <- lm(y ~ x)
  
  new <- data.frame(x = test_data[,"DC_mean"])
  pred_lm_p1 <- predict(model_lm, new)
  
  ## get the prediction performance
  p1 = postResample(test_data[,n_target], pred_lm_p1)
  predict_l <- data.frame(test_data[,n_target], pred_lm_p1)
  colnames(predict_l)=c(n_target,paste0("lm_predicted_",n_target))
  results<-list(p1,predict_l)
  return (results)
}

# function to remove outlier 
remove_outlier<-function(dataset,target){
  dataset<-subset(dataset,dataset[,target]!='NA')
  Q1=as.numeric(quantile(dataset[,target],c(0.25,0.75))[1])
  Q3=as.numeric(quantile(dataset[,target],c(0.25,0.75))[2])
  IQR=Q3-Q1
  upper_limit<-Q3+1.5*IQR
  lower_limit<-Q1-1.5*IQR
  new_data<-subset(dataset,(dataset[,target]<=upper_limit) & (dataset[,target]>=lower_limit))
  print(dim(new_data))
  return(new_data)
}

# combine previous functions to create daily nutrient concentrations 
generate_nutrient_daily <- function(data_set, Flow_DC, Rainfall, n_target) {
  
  print(n_target)
  ## select nutrient_target
  combined_target <- combined_nutrient_unscaled(data_set, Flow_DC, Rainfall, n_target)
  sampled_target_selected <- subset(combined_target,combined_target[,n_target]!='NA')
  ## remove the outliers
  sampled_target_selected<-remove_outlier(sampled_target_selected,n_target)
  
  combined_target$Date <- as.numeric(combined_target$Date)
  
  ## build the model_build
  set.seed(1234)
  rf <- model_build(sampled_target_selected, n_target)
  predict_rf_n_nutrient <- data.frame(combined_target["Date"], 
                                      as.data.frame(h2o::h2o.predict(rf,as.h2o(combined_target))[,"predict"]))
  names(predict_rf_n_nutrient) <- c("Date", paste0(n_target, "_p"))
  return(predict_rf_n_nutrient)
}  

# use generate_nutrient_daily to create daily nutrient concentration list 
generated_nutrient_list<-function(dataset, Flow_DC, Rainfall, n_target){
  
  if(n_target=="FRP"){
    nutrient_pool=c("DOC","TP","TN","DON","TSS")
  } else {
    nutrient_pool=c("DOC","TP","TN","DON")
  }  
  
  generated_nutrient<-nutrient_pool[!(nutrient_pool %in% n_target)]
  nutrient_list=vector(mode = "list",length = length(generated_nutrient))
  for(ii in 1:length(generated_nutrient)){
    if(length(subset(dataset[,generated_nutrient[ii]],dataset[,generated_nutrient[ii]]>0))<100) next
    
    g_nutrient<- generate_nutrient_daily(dataset, Flow_DC, Rainfall, generated_nutrient[ii])
    nutrient_list[[ii]]<-g_nutrient
  }
  return(nutrient_list)
}

# load the dataset 
Nutrient <- read.csv("PJ.csv", header = T)
Flow <- read.csv("flow.csv", header = T)

Rainfall <- read.csv("rainfall_raw.csv", header = T)
Rainfall<-create_date(Rainfall)

# rename the col
Nutrient$Collection.Device<-NULL
names(Nutrient) <- c("Date","DOC","DON","NOx","TN","NH4","TP","FRP","TSS")
names(Flow) <- c("Date","DC_mean")
names(Rainfall) <- c("Rainfall","Date")

# convert into date format 
Nutrient$Date<-lubridate::dmy(as.character(Nutrient$Date))
Flow$Date<-lubridate::ymd(as.character(Flow$Date))
Rainfall$Date<-lubridate::ymd(as.character(Rainfall$Date))

# convert into numeric formate 
for (i in seq(2,ncol(Nutrient),1)){
  Nutrient[,i]<-as.numeric(as.character(Nutrient[,i]))
}

# find date range of flow data 
time.min <- range(Flow$Date)[1]
time.max <- range(Flow$Date)[2]

Nutrient<-subset(Nutrient,(Nutrient$Date<=time.max)&(Nutrient$Date>=time.min))

# generate new variables 
# define DIN values 
Nutrient$DIN<-Nutrient[,"NH4"]+Nutrient[,"NOx"]

Rainfall[is.na(Rainfall$Rainfall),"Rainfall"]<-mean(Rainfall$Rainfall,na.rm=T)

new_flow<-data.frame(Date=seq(as.Date("1970/01/02"), as.Date("2018/07/20"), "day"))
new_flow<-merge(new_flow,Flow,by="Date",all.x=T)
new_flow[is.na(new_flow$DC_mean),"DC_mean"]=median(Flow$DC_mean)
Flow<-new_flow

# create dischargem rainfall features 
# lagged discharge in 3, 7, 15 days  
for (d in c(3,7,15)) {
  Flow[, paste0("mean", as.character(d))] <- rollmeanr(Flow$DC_mean, k = d, fill = 0)
}

# accumulated rainfall in 3, 7, 15 days  
for (d in c(3,7,15)) {
  Rainfall[, paste0("rsum", as.character(d))] <- rollsumr(Rainfall$Rainfall, k = d, fill = 0)
}

# max rainfall in 3, 7, 15 days  
for (d in c(3,7,15)) {
  Rainfall[, paste0("rmean", as.character(d))] <- rollmeanr(Rainfall$Rainfall, k = d, fill = 0)
}

Flow_DC<-Flow


for (n_target in all_n_target){
  
  final_results<-data.frame()
  predict_results<-data.frame()
  
  Nutrient2<-remove_outlier(Nutrient,n_target)
  Nutrient2$Date<-format(as.Date(Nutrient2$Date), "%Y-%m-%d")
  
  # get combined dataset
  All_combined<-combined_nutrient_unscaled(Nutrient2, Flow_DC, Rainfall, n_target)
  Nutrient_sampled <- subset(All_combined, All_combined[[n_target]] != 'NA')
  Nutrient_sampled<-Nutrient_sampled[order(Nutrient_sampled$Date),]
  
  set.seed(10)
  seed.list<-sample(1111:4000,500,replace =F)
  var_number=15
  
  # repeat the process for 30 times 
  for (tt in seq(1,30)){
    
    print(c("n_target=",n_target))
    print(tt)
    seed<-seed.list[tt]
    set.seed(seed)
    
    # split training and testing datasets 
    trainIndex <- createDataPartition(Nutrient_sampled[,n_target], p = 0.8, list = FALSE,groups=20)  
    training_p1 <- Nutrient_sampled[trainIndex,]
    testing_p1 <- Nutrient_sampled[-trainIndex,]
    
    names(testing_p1) <- colnames(training_p1)
    
    training_p1 <- round(training_p1, digits =7)
    testing_p1 <- round(testing_p1, digits = 7)
    
    train_date<-training_p1$Date
    test_date<-testing_p1$Date
    
    training_p1[,n_target]<-log10(training_p1[,n_target])
    testing_p1[,n_target]<-log10(testing_p1[,n_target])        
    
    max_target<-max(training_p1[,n_target])
    min_target<-min(training_p1[,n_target])
    
    s_data<-standardize(training_p1,testing_p1)
    training_p1<-s_data[[1]]
    testing_p1<-s_data[[2]]
    
    # create linear model 
    pred_lm_p1<-lm_model(training_p1,testing_p1,n_target)[[2]]
    pred_lm_p1[,1]<-standardize_reverse(pred_lm_p1[,1],max_target,min_target)
    pred_lm_p1[,2]<-standardize_reverse(pred_lm_p1[,2],max_target,min_target)
    pred_lm_p1[,1]<-10^(pred_lm_p1[,1])
    pred_lm_p1[,2]<-10^(pred_lm_p1[,2])
    
    # test linear model performance 
    p1<-postResample(pred_lm_p1[,2],pred_lm_p1[,1])
    
    training_p1$DC_mean<-NULL
    testing_p1$DC_mean<-NULL
        
    important_val=find_importVar(training_p1,testing_p1,n_target,var_number)
    training_p2<-important_val[[1]]
    testing_p2<-important_val[[2]]
    
    training_p2<-training_p2[order(training_p2$Date),] 
    testing_p2<-testing_p2[order(testing_p2$Date),]    
    
    if("DC_mean" %in% colnames(training_p2)){
      training_p2$DC_mean<-NULL
      testing_p2$DC_mean<-NULL 
    }
    
    # build one stage machine learning model 
    rf_p2<- model_build(as.h2o(training_p2), n_target)
    
    ## test in testing set
    testing_p2<-as.h2o(testing_p2)
    pred_rf_p2=as.data.frame(h2o::h2o.predict(rf_p2,testing_p2))
    
    ## get the prediction performance
    testing_p2<-as.data.frame(testing_p2)
    pred_rf_p2<-data.frame(testing_p2[,n_target],pred_rf_p2$predict)
    
    pred_rf_p2[,1]<-standardize_reverse(pred_rf_p2[,1],max_target,min_target)
    pred_rf_p2[,2]<-standardize_reverse(pred_rf_p2[,2],max_target,min_target)
    
    pred_rf_p2[,1]<-10^pred_rf_p2[,1]
    pred_rf_p2[,2]<-10^pred_rf_p2[,2]
    # test one stage machine learning model performance 
    p2 = postResample(pred_rf_p2[,2],pred_rf_p2[,1])
    
  
    Nutrient_training <- Nutrient2[trainIndex,]
    nutrient_data <- generated_nutrient_list(Nutrient_training,n_target)
        
    training_p3<-training_p2
    testing_p3<-testing_p2
    
    training_p3$Date<-train_date 
    testing_p3$Date<-test_date    
    # add generated daily nutrients as additional features 
    for (i in seq(1,length(nutrient_data))){
      if(length(nutrient_data[[i]])==0) next
      nutrient_data[[i]][2]<-log(abs(nutrient_data[[i]][2]))
      
      min_v<-min(nutrient_data[[i]][2])
      max_v<-max(nutrient_data[[i]][2])
      nutrient_data[[i]][2]<-(nutrient_data[[i]][2]-min_v)/(max_v-min_v) 
      
      training <- merge(nutrient_data[[i]], training_p3, by = "Date", all.y = T)
      testing <- merge(nutrient_data[[i]], testing_p3, by = "Date", all.y = T)
      
      training_p3 <- training
      testing_p3 <- testing
    }
    
    min_date<-min(training_p3[,"Date"])
    max_date<-max(training_p3[,"Date"])
    
    training_p3[,"Date"]<-(training_p3[,"Date"]-min_date)/(max_date-min_date)
    testing_p3[,"Date"]<-(testing_p3[,"Date"]-min_date)/(max_date-min_date)
    
    training_p3<-training_p3[order(training_p3$Date),] 
    testing_p3<-testing_p3[order(testing_p3$Date),]    
    
    set.seed(719)
    # build hybrid machine learning model 
    rf_p3 <- model_build(as.h2o(training_p3), n_target)
    
    testing_p3<-as.h2o(testing_p3)
    pred_rf_p3=as.data.frame(h2o::h2o.predict(rf_p3,testing_p3))
    testing_p3<-as.data.frame(testing_p3)
    
    pred_rf_p3<-data.frame(testing_p3[,n_target],pred_rf_p3$predict)
    testing_p3<-as.h2o(testing_p3)
    
    pred_rf_p3[,1]<-standardize_reverse(pred_rf_p3[,1],max_target,min_target)
    pred_rf_p3[,2]<-standardize_reverse(pred_rf_p3[,2],max_target,min_target)
    
    pred_rf_p3[,1]<-10^(pred_rf_p3[,1])
    pred_rf_p3[,2]<-10^(pred_rf_p3[,2])
    
    # test model performance 
    testing_p3<-as.data.frame(testing_p3)
    p3 = postResample(pred_rf_p3[,2],pred_rf_p3[,1])
    
    # format performance metrics 
    p1<-as.data.frame(t(p1))
    p1$MEF=MEF(pred_lm_p1[,2],pred_lm_p1[,1])
    colnames(p1)<-c("RMSE_p1","R2_p1","MAE_p1","MEF_p1")
    
    p2<-as.data.frame(t(p2))
    p2$MEF=MEF(pred_rf_p2[,2],pred_rf_p2[,1])
    colnames(p2)<-c("RMSE_p2","R2_p2","MAE_p2","MEF_p2")
    
    p3<-as.data.frame(t(p3))
    p3$MEF=MEF(pred_rf_p3[,2],pred_rf_p3[,1])
    colnames(p3)<-c("RMSE_p3","R2_p3","MAE_p3","MEF_p3")
    
    single_result<-data.frame(n_target,tt,p1,p2,p3)
    final_results<-rbind(final_results,single_result)
    rownames(final_results)<-NULL
    
    single_predict_result<-data.frame(n_target,tt,pred_lm_p1,pred_rf_p2[,2],pred_rf_p3[,2])
    predict_results<-rbind(predict_results,single_predict_result)
    rownames(predict_results)<-NULL
    
    print(n_target)
    print(p1)
    print(p2)
    print(p3)
    
  }
  write.csv(predict_results,paste0("PJ_",n_target,".csv"),row.names=FALSE)
} 
