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
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Deep learning 4j for one CNN framework runs over local spark.
 *
 */
public class DL4JRNNsparkTrain 
{
	private static final Logger log = LoggerFactory.getLogger(DL4JRNNsparkTrain.class);
	
    public static void main(String[] args) throws IOException, InterruptedException
    {

    	String TrainDataName = "scd_train540";
    	
        //Set parameter of RNN
    	int numEpoch = 100;        
    	int numInputs = 60;

        //6 classes (types of scd data). Classes have integer values 0, 1 or 2
        int numClasses = 6;
        int labelIndex = numInputs;
        int numOutNeuronsInRNNLayer = 500;

        int seed = 123;
        double learningRate = 0.1;
        double momentum = 0.9;       
        
        //Set parameter of TrainingMaster        
        int batchSize = 32;
        int averFrequency = 5;
        
        String TraindatacsvPath = "/root/demo_nfs/datasetDNN/" + TrainDataName + ".csv";        
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("Train RNN algorithm for " + TrainDataName);
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
        
        //The 2-RNN based on the two RNN Layers
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        		.seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(0.01)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Nesterovs(learningRate, momentum))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .list()
                .layer(0, new GravesLSTM.Builder()
                		.name("RNN1")
        				.nIn(numInputs)
        				.nOut(numOutNeuronsInRNNLayer)
        				.activation(Activation.RELU)
        				.build())
                .layer(1, new GravesLSTM.Builder()
                		.name("RNN2")
        				.nIn(numInputs)
        				.nOut(numOutNeuronsInRNNLayer)
        				.activation(Activation.RELU)
        				.build())
        		.layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        				.name("output")
        				.activation(Activation.SOFTMAX)        				
        				.nIn(numOutNeuronsInRNNLayer)
        				.nOut(numClasses)
        				.build())                
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
        
        StatsStorage statsStorage  = new FileStatsStorage(new File("/root/demo_nfs/sparkRNNTrainingStats.dl4j"));
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
        	log.info("RNN Model Completed Epoch {}", i);
        	MultiLayerNetwork sparkDL4Jnet= sparkDL4Jnetwork.fit(trainDataset);
            
            if(i == numEpoch-1) {
            	//Save model
            	log.info("Build and Save RNN model with numEpoch = " + numEpoch);        
                String modelsavePath = "/root/demo_nfs/Models/RNN_model";
                sparkDL4Jnet.save(new File(modelsavePath));
            }
            
        }
        
        log.info("****************RNN Model Built********************");
        sc.close();        
    }
    
}
