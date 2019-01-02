package com.itri.keras;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Deep learning 4j for importing keras model.
 *
 */
public class DL4J_spark_keras_iris {

	private static final Logger log = LoggerFactory.getLogger(DL4J_spark_keras_iris.class);
	
	public static void main(String[] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException, InterruptedException {
		
		String TestDataName = "iris_test";
		//Path of data source
        String TestdatacsvPath = "/root/demo_nfs/" + TestDataName + ".csv";
		
        String dnnResultPath = "/root/demo_nfs/DNNresult/";
		
		// label index
        int labelIndex = 4;
        // num of classes
        int numClasses = 3;
        // batchsize all
        int batchSize = 150;        
		
        //Set parameter of TrainingMaster        
        
        int averFrequency = 5;
        
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("spark://ubuntu7:7077");
        sparkConf.setAppName("Demo for predicting with CNN Model for " + TestDataName + " dataset");
        
        //Create one spark instance
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        
        
		 //Load csv file as a testing dataset
        log.info("Load iris dataset csv....");        
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
        
        log.info("Load keras_iris_test h5....");        
        MultiLayerNetwork networkConfig = KerasModelImport.importKerasSequentialModelAndWeights("/root/demo_nfs/keras_iris_test.h5");
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
        
        log.info("***** keras_spark test Evaluation *****");
        Evaluation evaluation = sparkDL4Jnetwork.evaluate(testDataset);        
        String evaluationStr = evaluation.stats().toString();        
        log.info("**********************evaluation status**********************\n" + evaluationStr);
        String resultName = TestDataName + "_keras_spark_test.txt";        
        FileUtils.writeStringToFile(new File(dnnResultPath + resultName), evaluationStr);        
        
        log.info("****************keras_spark Tested********************");        
        sc.close();
		
		

	}

}
