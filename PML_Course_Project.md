# Prediction with Human Activity Recognition Data
Charles Wylie - October 23, 2016 - Practical Machine Learning Course Project  




&nbsp;

### Project Background  

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is possible to collect a large amount of data about personal activity relatively inexpensively. People often quantify how much of a particular activity they do, but they rarely quantify how well they do it. We would like to determine if data from sensors attached to the body can be used to inform us whether an athlete is performing weightlifting exercises properly. Data for this project was collected from accelerometers on the belt, forearm, arm, and dumbell of six participants.  

The participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).  

The project goal is to predict the manner in which participants did the exercise (A, B, C, D, or E). This is the "classe" variable in the training set. This report will describe how we built our model, how we used cross validation, and what we think the expected out of sample error is. We will also use our prediction model to predict 20 different test cases.  

### Data Sources  

The training and testing data for this project are available at the following websites:  

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  

More information is available from the website http://groupware.les.inf.puc-rio.br/har  

Citation:  

*Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.*  

See also:  

*Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.*  


### Load and Process the Data  


```r
testing <- read.csv("pml-testing.csv", stringsAsFactors=F)
training <- read.csv("pml-training.csv", stringsAsFactors=F)
```

We attempted to use the nearZeroVar() function to identify zero covariates, however it did a poor job of flagging NA and empty variables. Inspection of the csv training file showed that many of the variables have no data or no use for the analysis. We excluded the first seven variables, and of the remainder, only the 52 features containing sensor measurements were kept in the training and testing datasets:  


```r
keep <- c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt", 
          "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", 
          "accel_belt_x", "accel_belt_y", "accel_belt_z", 
          "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", 
          "roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", 
          "gyros_arm_x", "gyros_arm_y", "gyros_arm_z", 
          "accel_arm_x", "accel_arm_y", "accel_arm_z", 
          "magnet_arm_x", "magnet_arm_y", "magnet_arm_z", 
          "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell", 
          "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", 
          "accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", 
          "magnet_dumbbell_x", "magnet_dumbbell_y", "magnet_dumbbell_z", 
          "roll_forearm", "pitch_forearm", "yaw_forearm", "total_accel_forearm", 
          "gyros_forearm_x", "gyros_forearm_y", "gyros_forearm_z", 
          "accel_forearm_x", "accel_forearm_y", "accel_forearm_z", 
          "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z", 
          "classe")

training <- training[,(names(training) %in% keep)]
testing_Quiz <- testing[,(names(testing) %in% keep)]
```

The testing dataset has been set aside until the final quiz. The training dataset will now be divided into a training set and validation set.  


```r
library(caret)

inTraining <- createDataPartition(training$classe, p = .75, list=FALSE)
training <- training[inTraining,]
testing <- training[-inTraining,]
```

### Build and Run the Model  

We set up the training run for x / y syntax for better performance with the large number of variables, and set a seed so that the model can be reproduced.  


```r
x <- training[,-53]
y <- training[,53]

set.seed(3737)
```

We run the 52 variable model using the random forest method (rf), with train control set for four-fold cross-validation (cv). This can take 10 minutes or longer.  


```r
library(randomForest)

modelFit52 <- train(x, y, data = training, method = "rf", 
        trControl = trainControl(method = "cv", number = 4))
```

We examine the model.  


```r
modelFit52
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 11039, 11038, 11038, 11039 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9905560  0.9880523
##   27    0.9904879  0.9879671
##   52    0.9854602  0.9816076
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

### Test the Model  

Now run the model on the validation dataset and examine the result using the confusionMatrix() function.  


```r
predict52 <- predict(modelFit52, testing)

confusionMatrix(predict52, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1045    0    0    0    0
##          B    0  710    0    0    0
##          C    0    0  642    0    0
##          D    0    0    0  605    0
##          E    0    0    0    0  696
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.2826    
##     P-Value [Acc > NIR] : < 2.2e-16 
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000    1.000   1.0000   1.0000   1.0000
## Specificity            1.0000    1.000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000    1.000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000    1.000   1.0000   1.0000   1.0000
## Prevalence             0.2826    0.192   0.1736   0.1636   0.1882
## Detection Rate         0.2826    0.192   0.1736   0.1636   0.1882
## Detection Prevalence   0.2826    0.192   0.1736   0.1636   0.1882
## Balanced Accuracy      1.0000    1.000   1.0000   1.0000   1.0000
```

The model has perfect accuracy 1.00 on the validation set, so we run the model on the quiz dataset to get the answer we will report in the quiz.  


```r
predict(modelFit52, testing_Quiz)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

### Reduce the Number of Variables in the Model

We use the *varImp()* function to find the most important predictors to keep in our model. We hope to keep our out-of-sample error rate at 0.01 or less while using fewer predectors. We will select the seven variables with the highest score:  


```r
(keep <- varImp(modelFit52))
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                   Overall
## roll_belt          100.00
## yaw_belt            80.88
## magnet_dumbbell_z   69.46
## pitch_belt          65.88
## pitch_forearm       65.28
## magnet_dumbbell_y   64.97
## magnet_dumbbell_x   55.17
## roll_forearm        53.97
## accel_belt_z        47.40
## roll_dumbbell       46.75
## accel_dumbbell_y    46.36
## magnet_belt_y       45.23
## magnet_belt_z       42.71
## accel_dumbbell_z    39.02
## roll_arm            36.59
## accel_forearm_x     32.48
## accel_dumbbell_x    31.22
## yaw_dumbbell        30.60
## gyros_belt_z        30.12
## magnet_arm_x        29.99
```

```r
# keep <- keep$importance
# keep <- c(rownames(keep)[keep > 25], "classe")
```

The following predictors (plus the classe outcome variable) will be used to fit our model:  


```r
keep <- c("roll_belt", "pitch_belt", "yaw_belt", "roll_forearm", "pitch_forearm", "magnet_dumbbell_y", "magnet_dumbbell_z", "classe")
```




Reduce the datasets and check their dimensions.


```r
training <- training[,(names(training) %in% keep)]
testing <- testing[,(names(testing) %in% keep)]
testing_Quiz <- testing_Quiz[,(names(testing_Quiz) %in% keep)]

dim(training); dim(testing); dim(testing_Quiz)
```

```
## [1] 14718     8
```

```
## [1] 3702    8
```

```
## [1] 20  7
```

### Run the Random Forest Model on the Reduced Training Set


```r
#x <- training[,-8]
#y <- training[,8]

set.seed(3737)

modelFit7 <- train(classe ~ ., data = training, method = "rf", 
        trControl = trainControl(method = "cv", number = 4))
```

We examine the model.  


```r
modelFit7
```

```
## Random Forest 
## 
## 14718 samples
##     7 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 11039, 11038, 11038, 11039 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##   2     0.9838973  0.9796357
##   4     0.9836936  0.9793772
##   7     0.9777145  0.9718151
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

### Run the Model on the Reduced Validation Set 

Now run the model on the seven-predictor validation set and examine the result using the confusionMatrix() function.  


```r
predict7 <- predict(modelFit7, testing)
confusionMatrix(predict7, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1060    0    0    0    0
##          B    0  695    0    0    0
##          C    1    1  643    0    0
##          D    1    0    0  598    0
##          E    0    0    0    0  703
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9992          
##                  95% CI : (0.9976, 0.9998)
##     No Information Rate : 0.2869          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.999           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9981   0.9986   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   0.9993   0.9997   1.0000
## Pos Pred Value         1.0000   1.0000   0.9969   0.9983   1.0000
## Neg Pred Value         0.9992   0.9997   1.0000   1.0000   1.0000
## Prevalence             0.2869   0.1880   0.1737   0.1615   0.1899
## Detection Rate         0.2863   0.1877   0.1737   0.1615   0.1899
## Detection Prevalence   0.2863   0.1877   0.1742   0.1618   0.1899
## Balanced Accuracy      0.9991   0.9993   0.9997   0.9998   1.0000
```

Again the model has accuracy 1 or nearly 1 on the validation set, so we run the model on the quiz dataset to see if we get the same answer we reportd with the 52 predictor model.


```r
predict(modelFit7, testing_Quiz)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

We get the same results, so we are probably safe using as few as seven variables to predict the manner in which participants did the exercise. Though not shown here, attempts with six, four, and three predictors were tried, with decreasing levels of accuracy, and predictions on the quiz set that did not match the 52 variable model outcome.

### Conclusion

To build our model, we examined the training set and eliminated variables that were not sensor measurements or did not contain data. We set aside the 20 quiz cases, then cut 25% out of the training set to use for testing our model, and used 75% for training. We used a random forest training model with four-fold cross validation.  

We expect the out of sample error rate to be nearly 0.00% using any number of predictors more than seven. Using three predictors the error rate was 1.3% (accuracy = 0.9872), however five of the twenty quiz case results did not match our prediction. Using four and six variable models, the error rate was 0.05% on the testing set, and one prediction did not match in the quiz set.

Using 52 variables, our prediction for the 20 quiz cases is B A B A A E D B A A B C B A E E A B B B.  

&nbsp;

