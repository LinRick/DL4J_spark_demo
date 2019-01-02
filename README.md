# DL4J-spark-demo project
- Deeplearning Exercise
- Run DL4J including CNN and RNN algorithms over Spark platform (Rick)

## Description
- There are two algorithms for demoing
- 
    **Including program:**

    | program name | program description |
    | ------ | ------ |
    |DL4JCNNsparkTrain| CNN algorithm for training phase|
    |DL4JCNNsparkTest| CNN algorithm for testing phase|
    |DL4JRNNsparkTrain| RNN algorithm for training phase|
    |DL4JRNNsparkTest| RNN algorithm for testing phase|
    |DL4J_keras_iris| import model built by keras to predicting new data without spark platform|
    |DL4J_spark_keras_iris| import model built by keras to predicting new data over spark|

### In the used CNN algorithm
> **Note:** 
The LeNet NN(Neural Network) based on the two CNN Layers and two pooling Layers
```sh
$ cd DL4J_spark_demo
$ mvn clean package
$ cp /target/DL4Jsparkdemo-0.1-shaded.jar /root/
# Using DL4Jsparkdemo-0.1-shaded.jar to run over spark under the multi-workers mode (standalone).
# Example: Building the LeNet NN (Training)
$./spark-submit --class com.itri.spark.DL4JCNNsparkTrain/root/demo_nfs/DL4J-spark-demo-0.1-shaded.jar
```
### In the used RNN algorithm
> **Note:** 
The RNN(Neural Network) based on the two RNN Layers
```sh
$ cd DL4J_spark_demo
$ mvn clean package
$ cp /target/DL4Jsparkdemo-0.1-shaded.jar /root/
# Using DL4Jsparkdemo-0.1-shaded.jar to run over spark under the multi-workers mode (standalone).
# Example: Building the RNN (Training)
$./spark-submit --class com.itri.spark.DL4JRNNsparkTrain/root/demo_nfs/DL4J-spark-demo-0.1-shaded.jar
```

### Import keras model in DL4J framework
> **Note:** 
The model in keras setting:

model = Sequential()

model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))

model.add(Dense(10, activation='relu', name='fc2'))

model.add(Dense(3, activation='softmax', name='output'))

optimizer = Adam(lr=0.001)

model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```sh
$ cd DL4J_spark_demo
$ mvn clean package
$ cp /target/DL4Jsparkdemo-0.1-shaded.jar /root/
# Using DL4Jsparkdemo-0.1-shaded.jar to run over spark under the multi-workers mode (standalone).
# Example: Predicting data
$./spark-submit --class com.itri.keras.DL4J_spark_keras_iris/root/demo_nfs/DL4J-spark-demo-0.1-shaded.jar
```