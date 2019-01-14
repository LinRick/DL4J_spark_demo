package com.itri.spark;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Deep learning 4j for training CNN framework runs over spark standalone mode.
 *
 */
public class DL4JCNNsparkTrain 
{
	private static final Logger log = LoggerFactory.getLogger(DL4JCNNsparkTrain.class);
	
    public static void main(String[] args) throws IOException, InterruptedException
    {
    	String TrainDataName = "mnist_train10000";

        //Set parameter of LeNet NN(Neural Network)
    	int numEpoch = 10;        
        int numInputs = 784;
        //10 classes (classes of data). Classes must be integer values {0,1,2,...,9}
        //Labels are the (numInputs+1)th value in each row
        int numClasses = 10;
        int labelIndex = numInputs;        
        int numOutNeuronsInCNNLayer1 = 20;
        int numOutNeuronsInCNNLayer2 = 50;        
        int numOutNeuronsInDenseLayer = 500;        
        int outHeightCNN = 28;
    	int outWidthCNN = 28;    	
    	
    	// nChannels = 1; only for black and white figure
        int nChannels = 1;         
        int seed = 123;
        double learningRate = 0.1;
        double momentum = 0.9;

        //Set parameter of TrainingMaster        
        int batchSize = 32;
        int averFrequency = 5;
   	    	
        //Path of data source
    	String TraindatacsvPath = "/root/demo_nfs/datasetDNN/" + TrainDataName + ".csv";

        //Set configures of spark
        SparkConf sparkConf = new SparkConf();        
        sparkConf.setMaster("spark://ubuntu7:7077");        
        sparkConf.setAppName("Demo for training CNN with " + TrainDataName + " dataset");

        //Create one spark instance
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        
        //Load csv file as a training dataset
        log.info("Load training data csv....");
        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new File(TraindatacsvPath)));        
        DataSetIterator Trainiterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
        	   .classification(labelIndex, numClasses)        		
        	   .build();        
        List<DataSet> trainData = new ArrayList<>();
        while(Trainiterator.hasNext()) trainData.add(Trainiterator.next());
        
        //Convert List<DataSet> to spark.RDD
        JavaRDD<DataSet> trainDataset = sc.parallelize(trainData);

        //The LeNet NN(Neural Network) based on the two CNN Layers and two pooling Layers 
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(0.01)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Nesterovs(learningRate, momentum))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .list()
                //first layer: 5*5 convolution layer
                .layer(0, new ConvolutionLayer.Builder(5,5)
               	     //nIn and nOut specify depth. 
               		 //nIn here is the nChannels and nOut is the number of filters to be applied
                		.name("cnn1")                		
               		 	.stride(1,1)
               		 	.nIn(nChannels)                        
                        .nOut(numOutNeuronsInCNNLayer1)
                        .activation(Activation.RELU)
                        .build())
                //second layer: 2*2 subsampling layer (Max-Pooling)
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                		.name("sampling1")
                		.kernelSize(2,2)
                        .stride(2,2)
                        .build())
                //third layer: 5*5 convolution layer
                .layer(2, new ConvolutionLayer.Builder(5,5)
                        //Note that nIn need not be specified in later layers
                		.name("cnn2")
                		.stride(1, 1)
                        .nOut(numOutNeuronsInCNNLayer2)
                        .activation(Activation.RELU)
                        .build())
                //4-th layer: 2*2 subsampling layer (Max-Pooling)
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                		.name("sampling2")
                		.kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder()
                		.activation(Activation.RELU)
                        .nOut(numOutNeuronsInDenseLayer)
                        .build())
                //5-th 還原image layer:
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                		.name("output")
                		.nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                //height height of the inputwidth Width of the inputdepth Depth, or number of channels
                .setInputType(InputType.convolutionalFlat(outHeightCNN,outWidthCNN,nChannels))                 
                .backprop(true).pretrain(false)
                .build();

        //Import the configured LeNet NN into MultiLayerNetwork
    	MultiLayerNetwork networkConfig = new MultiLayerNetwork(conf);
    	//MultiLayerNetwork initialization
    	networkConfig.init(); 	
    	
        //Create TrainingMaster of DL4J instance        
        //int examplesPerDataSetObject = 1;
        TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)                
        		 .rddTrainingApproach(RDDTrainingApproach.Direct)    
        		 .saveUpdater(true)        		 
        		 .averagingFrequency(averFrequency)
        		 .batchSizePerWorker(batchSize)        		 
                 .build();
        
        //Create DL4jMultiLayer instance over Spark with networkConfig and trainingMaster settings
        SparkDl4jMultiLayer sparkDL4Jnetwork = new SparkDl4jMultiLayer(sc,networkConfig,trainingMaster);
        
        StatsStorage statsStorage  = new FileStatsStorage(new File("/root/demo_nfs/sparkCNNTrainingStats.dl4j"));
        //StatsStorage statsStorage = new InMemoryStatsStorage();

        //ScoreIterationListener 每跌代n次, 便輸出一次損失函式的值
        sparkDL4Jnetwork.setListeners(statsStorage, 
        		Collections.singletonList(new StatsListener(null)));       
        
        //Set UI server instance to show records of learning
      	UIServer uiServer = UIServer.getInstance();

      	//Set UI server instance into MultiLayerNetwork
      	uiServer.attach(statsStorage);
        
        for (int i = 0; i < numEpoch; i++) {                	
        	//Bulid Model
        	log.info("CNN Model Completed Epoch {}", i);
        	MultiLayerNetwork sparkDL4Jnet= sparkDL4Jnetwork.fit(trainDataset);            

            if(i == numEpoch-1) {
            	//Save model
            	log.info("Build and Save CNN model with numEpoch = " + numEpoch);        
                String modelsavePath = "/root/demo_nfs/Models/CNN_model";
                sparkDL4Jnet.save(new File(modelsavePath));
            }
            
        }
        
        log.info("****************CNN Model Built********************");
        sc.close();
        
    }
    
}
