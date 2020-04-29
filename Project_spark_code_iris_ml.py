import pyspark
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark import SparkFiles
import os
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
import sys


"""
----------------------------------------------------------------------------
CREATE SPARK CONTEXT
CREATE SQL CONTEXT
----------------------------------------------------------------------------
"""
sc =SparkContext()
sc.setLogLevel("WARN")
sqlContext = SQLContext(sc)

#print runtime versions
print ('**********************')
print ('Python version: {}'.format(sys.version))
print ('Spark version: {}'.format(pyspark.version))
print('***********************')

"""
----------------------------------------------------------------------------
LOAD IRIS DATA
----------------------------------------------------------------------------
"""
data_dir="/home/gext/ismail.haris/BigDataHadoopSpark/projet-spark/Project/"
file = os.path.join(data_dir,"iris.csv")
panda_df = pd.read_csv(file)

iris_df=sqlContext.createDataFrame(panda_df)
iris_df.printSchema()

"""
------------------------------------------------------------------------------
EXPLORATORY DATA ANALYSIS
-----------------------------------------------------------------------------
-"""

#The Iris dataset is multivariate meaning there is more than one independent variable. 
#So we will carry out a basic multivariate EDA on it. 

#Description of iris dataset statistics
print ("Description du dataset")
iris_df.describe().show()
print()
iris_df.head()


#Show first 10 lines of dataset
print("Show 10 first lines")
iris_df.show(10)


#Show the number of each variety
print("Varieties and total number of each variaty")
iris_df.groupBy("variety").count().show()

#get the dimensions of the data 
print("Iris dataset dimensions")
print ( iris_df.count(), len(iris_df.columns))

#Missing values count 
print("Missing values")
data_agg = iris_df.agg(*[count(when(isnull(c),c)).alias(c) for c in iris_df.columns])
data_agg.show()

#See below for the correlation, since we need a vector as input ! 

print ("\n".join(["Early insights :",
       " *150 rows",
       " *4 independent variables to act as features",
        "*All have the same units of measurements (cm)",
       " *No missing data",
       " *3 target labels",
       " *No class imbalance (all the target classes have equal number of rows 50) "," "," "," "]))


"""
--------------------------------------------------------------------------------
DATA PROCESSING PIPELINE DEFINITION
--------------------------------------------------------------------------------
"""

"""--------------------------------------------------------------------------------
Spark Pipeline is a sequence of stages (Transformer, Estimator) : String indexer to
convert categorical data to numerics, standard scaler, vector assembler to put 
features in a dense vector, PCA features reduction, label one hot encoding. 
-----------------------------------------------------------------------------------
"""
from pyspark.ml import Pipeline

""" STRING INDEXER (Estimator) """ 

print("String Indexer")
#Add a numeric indexer for the label/target column 
stringIndexer = StringIndexer(inputCol="variety", outputCol="label")
si_model = stringIndexer.fit(iris_df)
irisNormDf = si_model.transform(iris_df)
irisNormDf.printSchema()
irisNormDf.select("variety", "label").distinct().show()
irisNormDf.select("variety", "label").distinct().collect()


"""Vector Assembler (Transformer)"""

print ("Vector Assembler")
#Transform to a DataFrame for input to Machine Learning. 
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = ["sepal_length","sepal_width", "petal_length","petal_width"], outputCol = "features")
irisLpDf = assembler.transform(irisNormDf)
#from pyspark.ml.linalg import Vectors
#def transformToLabeledPoint(row):
#    lp = (row["variety"], row["ind_variety"],\
#            Vectors.dense([row["sepal_length"],\
#            row["sepal_width"],\
#            row["petal_length"],\
#            row["petal_width"]]))
#    return lp
#irisLp = irisNormDf.rdd.map(transformToLabeledPoint)
#irisLpDf = sqlContext.createDataFrame(irisLp, ["species", "label","features" ])
#irisLpDf.show(10)
#irisLpDf.printSchema()

"""Explore the Correlation between features""" 

print ("Correlation matrices")
from pyspark.ml.stat import Correlation
#linear correlation
r = Correlation.corr(irisLpDf, "features").head()
print ("Pearson correlation matrix:\n" + str(r[0]))
#monotonic correlation
r2 = Correlation.corr(irisLpDf,"features","spearman").head()
print("Spearman correlation matrix\n"+str(r2[0]))

"""Standard Scaler(Estimator) """

print("\n".join(["Les features sont tres correles entre eux sauf sepal width qui n'a pas\
l'air d'etre correle avec les autres. +80% de correlation entre tous les autres, moins de \
50% entre sepal width et les autres features (sepal length, petal length et petal width)"," "," "," "," ","Standard Scaler"]))
#We will use the StandardScaler to normalize each feature to have unit standard deviation and 0 mean. 

from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import StandardScaler
scaler = StandardScaler (inputCol = "features", outputCol = "scaled_features",withStd=True, withMean = False)
scalerModel = scaler.fit(irisLpDf)
irisNormalizedDf = scalerModel.transform(irisLpDf)
irisNormalizedDf.printSchema()



""" Label OneHotEncoder (Estimator \ the Transformer is now deprecated)"""

print("One Hot Encoder")
from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(dropLast = False,  inputCol = "label", outputCol = "encoded_label")
encoder_model = encoder.fit(irisNormalizedDf)
irisNormalizedencodedDf = encoder_model.transform(irisNormalizedDf)
irisNormalizedencodedDf.printSchema()


""" PCA Principal Component Analysis"""

print ("PCA Analysis")
from pyspark.ml.feature import PCA
pca = PCA(k = 3, inputCol = "scaled_features", outputCol = "pcaFeatures")
pca_model = pca.fit(irisNormalizedencodedDf)
iris_pcaDf = pca_model.transform(irisNormalizedencodedDf)
iris_pcaDf.printSchema()
print ("PCA model explained variance", pca_model.explainedVariance,"Cumulative explained variance", pca_model.explainedVariance[0]+pca_model.explainedVariance[1]+pca_model.explainedVariance[2] )



""" Define the final DataFrame : apply the pipeline"""

print ("\n".join([" "," ","Data processing Pipeline"]))
pipeline = Pipeline(stages = [stringIndexer, assembler, scaler, encoder, pca])
(train,test) = iris_df.randomSplit([0.85,0.15])
pipeline_model = pipeline.fit(train)
iris_pipelineDf = pipeline_model.transform(iris_df)
iris_pipelineDf.printSchema()

#drop columns that are not required and keep only the variety the encoded label and the PCA features
print ("Final dataset ready for Machine Learning")
irisfinalDf = iris_pipelineDf.select("variety","label","encoded_label","pcaFeatures")
irisfinalDf.show(10)
irisfinalDf.printSchema()

#Split into training and testing data
(trainingData, testData) = irisfinalDf.randomSplit([0.85, 0.15],1234)
print("train set ",trainingData.count())
print("test set",testData.count())
testData.collect()

"""
------------------------------------------------------------------------
PERFORM MACHINE LEARNING
------------------------------------------------------------------------
"""

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


"""DECISION TREE CLASSIFIER"""

#We will use cross validation to find the optimal hyperparameters
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

dtParamGrid = ParamGridBuilder()\
        .addGrid(DecisionTreeClassifier.maxDepth,[5,10,20,50,100])\
        .build()

dtCrossval = CrossValidator(estimator = DecisionTreeClassifier(impurity ="gini", labelCol="label",featuresCol="pcaFeatures"),
                            estimatorParamMaps = dtParamGrid, 
                            evaluator=MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label",metricName="accuracy"), 
                            numFolds = 5)

#create the model
#dtClassifier = DecisionTreeClassifier(maxDepth=4, labelCol="label",featuresCol="pcaFeatures")
import time
dt_start = time.time()
dtModel = dtCrossval.fit(trainingData)
dt_end = time.time()
print("Decision Tree Classifier")
print()
print()
#print("NumNodes", dtModel.numNodes)
#print("Depth", dtModel.bestModel.getEstimatorParamMap()[np.argmax(dtModel.avgMetrics)])


#Predict on the test data
dtpredictions = dtModel.transform(testData)
dtpredictions.select("prediction", "variety", "label").collect()

#evaluate accuracy 
print("Decision Tree Classifier Accuracy evaluation")
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label", metricName="accuracy")
print (evaluator.evaluate(dtpredictions))


#Draw a confusion matrix
print("Confustion Matrix")
dtpredictions.groupBy("label","prediction").count().show()

"-----------------------------------------------------------------------------"

"""RANDOM FOREST  CLASSIFIER"""

from pyspark.ml.classification import RandomForestClassifier
#We will use cross validation to find the optimal hyperparameters
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

rfParamGrid = ParamGridBuilder()\
        .addGrid(RandomForestClassifier.numTrees,[5,10,50,30,100,500,1000])\
        .addGrid(RandomForestClassifier.maxDepth,[5,10,20,50,100])\
        .build()

rfCrossval = CrossValidator(estimator = RandomForestClassifier(impurity = "gini", labelCol="label",featuresCol="pcaFeatures"), estimatorParamMaps = rfParamGrid, evaluator=MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label",metricName="accuracy"), numFolds = 5)

#create the model
import time
rf_start = time.time()
rfModel = rfCrossval.fit(trainingData)
rf_end = time.time()
print("Random Forest Classifier")
print()
print()

#Predict on the test data
rfpredictions = rfModel.transform(testData)
rfpredictions.select("prediction", "variety", "label").collect()

#evaluate accuracy 
print("Random Forest Classifier Accuracy evaluation")
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label", metricName="accuracy")
print (evaluator.evaluate(rfpredictions))

#Draw a confusion matrix
print("Random Forest Confustion Matrix")
rfpredictions.groupBy("label","prediction").count().show()

"""-------------------------------------------------------------------------------------------------"""

"""Gradient Boosted tree classifier"""

print("Gradient Boosted Tree Classifier")
#from pyspark.ml.classification import GBTClassifier
#We will use cross validation to find the optimal hyperparameters
#from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

#gbtParamGrid = ParamGridBuilder()\
#        .addGrid(GBTClassifier.maxIter,[5,10,50,100])\
#        .build()

#gbtCrossval = CrossValidator(estimator = GBTClassifier(labelCol="label",featuresCol="pcaFeatures"), estimatorParamMaps = gbtParamGrid, evaluator=MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label",metricName="accuracy"), numFolds = 5)

#create the model
#import time
#gbt_start = time.time()
#gbtModel = gbtCrossval.fit(trainingData)
#gbt_end = time.time()
#print("Gradient Boosted Tree Classifier")
#print()
#print()

#Predict on the test data
#gbtpredictions = gbtModel.transform(testData)
#gbtpredictions.select("prediction", "variety", "label").collect()

#evaluate accuracy 
#print("Gradient Boosted Tree Classifier Accuracy evaluation")
#evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label", metricName="accuracy")
#print (evaluator.evaluate(gbtpredictions))

#Draw a confusion matrix
#print("Gradient Boosted Tree Confustion Matrix")
#gbtpredictions.groupBy("label","prediction").count().show()


print ("THE PROBLEM WITH GRADIENT BOOSTED TREE CLASSIFIER IS THAT FOR NOW THE spark.ml IMPLEMENTATION SUPPORTS BINARY CLASSIFICATION SO WE CANNOT USE IT HERE SINCE WE HAVE 3 TARGET CLASSES")



"""-------------------------------------------------------------------------------------------------"""

"""MULTILAYER PERCEPTRON MLP CLASSIFIER """

from pyspark.ml.classification import MultilayerPerceptronClassifier
#We will use cross validation to find the optimal hyperparameters
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
#mlpParamGrid = ParamGridBuilder()\
#        .addGrid(MultilayerPerceptronClassifier.maxIter,[100, 200, 500])\
#        .addGrid(MultilayerPerceptronClassifier.blockSize,[10,20,30])\
#        .addGrid(MultilayerPerceptronClassifier.layers, [[3,6,6,3],[3,20,20,3],[3,100,100,3]])\
#        .build()
layers = [3,20,20,3]
#mlpCrossval = CrossValidator(estimator=MultilayerPerceptronClassifier(layers=layers,labelCol="label",featuresCol="pcaFeatures", solver = "l-bfgs", seed = 1234), estimatorParamMaps = mlpParamGrid, evaluator=MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label",metricName="accuracy"), numFolds = 5)
mlp = MultilayerPerceptronClassifier(blockSize = 10, layers = layers,labelCol="label",
                                     featuresCol="pcaFeatures", solver = "l-bfgs", seed = 1234)
#create the model
import time
mlp_start = time.time()
mlpModel = mlp.fit(trainingData)
mlp_end = time.time()
print("MLP Classifier")
print()
print()

#Predict on the test data
mlppredictions = mlpModel.transform(testData)
mlppredictions.select("prediction", "variety", "label").collect()

#evaluate accuracy 
print("MLP Classifier Accuracy evaluation")
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label", metricName="accuracy")
print (evaluator.evaluate(mlppredictions))

#Draw a confusion matrix
print("MLP Confustion Matrix")
mlppredictions.groupBy("label","prediction").count().show()



"""------------------------------------------------------------------------------------------------"""

"""Logistic Regression CLASSIFIER """

from pyspark.ml.classification import LogisticRegression
#We will use cross validation to find the optimal hyperparameters
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

lrParamGrid = ParamGridBuilder()\
        .addGrid(LogisticRegression.regParam,[0.01,0.001,0.1, 0.5])\
        .addGrid(LogisticRegression.elasticNetParam,[0,0.2,0.5,0.8,1])\
        .build()
#   .addGrid(LogisticRegression.maxIter,[10, 100])\
lrCrossval = CrossValidator(estimator = LogisticRegression(labelCol="label",featuresCol="pcaFeatures"), estimatorParamMaps = lrParamGrid, evaluator=MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label",metricName="accuracy"), numFolds = 5)
#lr = LogisticRegression(maxIter = 100, regParam = 0.01, elasticNetParam = 0.8,labelCol="label",featuresCol="pcaFeatures")
#create the model
import time
lr_start = time.time()
lrModel = lrCrossval.fit(trainingData)
lr_end = time.time()
print("Logistic  Classifier")
print()
print()

#Predict on the test data
lrpredictions = lrModel.transform(testData)
lrpredictions.select("prediction", "variety", "label").collect()

#evaluate accuracy 
print("Logistic  Classifier Accuracy evaluation")
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label", metricName="accuracy")
print (evaluator.evaluate(lrpredictions))

#Draw a confusion matrix
print("Logistic Confustion Matrix")
lrpredictions.groupBy("label","prediction").count().show()


"""-------------------------------------------------------------------------------------------------"""

"""SVM CLASSIFIER"""

print("SVM Classifier")
"""
from pyspark.ml.classification import LinearSVC
#We will use cross validation to find the optimal hyperparameters
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

svmParamGrid = ParamGridBuilder()\
        .addGrid(LinearSVC.maxIter,[10, 100, 500])\
        .addGrid(LinearSVC.regParam,[0.1,0.3,0.5,0.8,])\
        .build()

svmCrossval = CrossValidator(estimator = LinearSVC(labelCol="label",featuresCol="pcaFeatures"), estimatorParamMaps = svmParamGrid, evaluator=MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label",metricName="accuracy"), numFolds = 5)

#create the model
import time
svm_start = time.time()
svmModel = svmCrossval.fit(trainingData)
svm_end = time.time()
print("SVM  Classifier")
print()

#Predict on the test data
svmpredictions = svmModel.transform(testData)
svmpredictions.select("prediction", "variety", "label").collect()

#evaluate accuracy 
print("SVM Classifier Accuracy evaluation")
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label", metricName="accuracy")
print (evaluator.evaluate(svmpredictions))

#Draw a confusion matrix
print("SVM Confustion Matrix")
svmpredictions.groupBy("label","prediction").count().show()
"""
print ("THE PROBLEM WITH SVM CLASSIFIER IS THAT FOR NOW THE spark.ml IMPLEMENTATION SUPPORTS BINARY CLASSIFICATION SO WE CANNOT USE IT HERE SINCE WE HAVE 3 TARGET CLASSES")

"""
---------------------------------------------------------------------------------------------------
COMPARISION OF PERFORMANCE
---------------------------------------------------------------------------------------------------
"""

list_perf = [(0,"Decision Tree",dt_end-dt_start,evaluator.evaluate(dtpredictions)),(1,"Random Forest", rf_end-rf_start,evaluator.evaluate(rfpredictions)),(2,"Logistic Regression", lr_end-lr_start,evaluator.evaluate(lrpredictions)),(3,"SVM", "None","None"),(4,"MLP",mlp_end-mlp_start,evaluator.evaluate(mlppredictions)), (5,"Gradient Boosting Tree", "None", "None")]
rdd = sc.parallelize(list_perf)
ppl_rdd = rdd.map(lambda x: Row(Index = x[0], ML_Algorithm = x[1], Fit_time = x[2], Accuracy = x[3]))
df = sqlContext.createDataFrame(ppl_rdd)
df.show()



#[(0,"Decision Tree",dt_end-dt_start),(1,"Random Forest", rd_end-rf_start),(2,"Logistic Regression", lr_end-lr_start),(3,"SVM", "None"),(4,"MLP","None")],["Index","ML Algorithm", "Fit Ti]


print ("Logistic Regression is clearly faster to fit than the others algorithms. MLP and Logistic Regression offer the maximum accuracy ")

















