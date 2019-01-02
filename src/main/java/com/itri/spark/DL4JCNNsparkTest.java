package com.itri.spark;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Deep learning 4j for one CNN framework runs over local spark.
 *
 */
public class DL4JCNNsparkTest 
{
	private static final Logger log = LoggerFactory.getLogger(DL4JCNNsparkTest.class);
	
    public static void main(String[] args) throws IOException, InterruptedException
    {    	

    	String TestDataName = "mnist_test100";        
        int numInputs = 784;
        
        //10 classes (classes of data). Classes must be integer values {0,1,2,...,9}
        //Labels are the (numInputs+1)th value in each row
        int numClasses = 10;
        int labelIndex = numInputs;   
        
        //Set parameter of TrainingMaster        
        int batchSize = 32;
        int averFrequency = 5;
        
        //Path of data source
        String TestdatacsvPath = "/root/demo_nfs/datasetDNN/" + TestDataName + ".csv";        

        //Path of results of learning  
        String dnnResultPath = "/root/demo_nfs/DNNresult/";
    	
    	File directory = new File(dnnResultPath);
        if (! directory.exists()){
            boolean isDirectoryCreated = directory.mkdir();
            if(isDirectoryCreated)
            	System.out.println("Directory created successfully");
            else
                System.out.println("Directory was not created successfully");
        }
        
        
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("spark://ubuntu7:7077");
        sparkConf.setAppName("Demo for predicting with CNN Model for " + TestDataName + " dataset");
        
        //Create one spark instance
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
          
        //Load csv file as a testing dataset
        log.info("Load testing data csv....");        
        int numLinesToSkip = 0;
        char delimiter = ',';      
        RecordReader testReader = new CSVRecordReader(numLinesToSkip, delimiter);
        testReader.initialize(new FileSplit(new File(TestdatacsvPath)));
        DataSetIterator Testiterator = new RecordReaderDataSetIterator.Builder(testReader, batchSize)
     		   .classification(labelIndex, numClasses)        		
     	       .build();
        List<DataSet> testData = new ArrayList<>();
        while(Testiterator.hasNext()) testData.add(Testiterator.next());        
        
        //Convert List<DataSet> to spark.RDD
        JavaRDD<DataSet> testDataset = sc.parallelize(testData);
                
        log.info("Load CNN model....");
        String modelLoadPath = "/root/demo_nfs/Models/CNN_model";
        MultiLayerNetwork networkConfig = ModelSerializer.restoreMultiLayerNetwork(modelLoadPath);        
        networkConfig.init();
        
        //Create TrainingMaster of DL4J instance
        TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)                
        		 .rddTrainingApproach(RDDTrainingApproach.Direct)    
        		 .saveUpdater(true)        		 
        		 .averagingFrequency(averFrequency)
        		 .batchSizePerWorker(batchSize)        		 
                 .build();
        
        //Create DL4jMultiLayer instance over Spark with networkConfig and trainingMaster settings
        SparkDl4jMultiLayer sparkDL4Jnetwork = new SparkDl4jMultiLayer(sc, networkConfig, trainingMaster);
        
        log.info("***** CNN Model Evaluation *****");
        Evaluation evaluation = sparkDL4Jnetwork.evaluate(testDataset);        
        String evaluationStr = evaluation.stats().toString();        
        log.info("**********************evaluation status**********************\n" + evaluationStr);
        String resultName = TestDataName + "_CNN_result.txt";        
        FileUtils.writeStringToFile(new File(dnnResultPath + resultName), evaluationStr);        
        
        log.info("****************CNN Model Tested********************");        
        sc.close();
        
    }
    
}
