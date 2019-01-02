package com.itri.keras;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
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
public class DL4J_keras_iris {

	private static final Logger log = LoggerFactory.getLogger(DL4J_keras_iris.class);
	
	public static void main(String[] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException, InterruptedException {
		
		String TestDataName = "iris_test";
		//Path of data source
        String TestdatacsvPath = "/root/demo_nfs/" + TestDataName + ".csv";
		
        log.info("Load keras_iris_test h5....");          
        
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("/root/demo_nfs/keras_iris_test.h5");	
		
		// label index
        int labelIndex = 4;
        // num of classes
        int numClasses = 3;
        // batchsize all
        int batchSize = 150;        
		
		 //Load csv file as a testing dataset
        log.info("Load iris dataset csv....");        
        int numLinesToSkip = 0;
        char delimiter = ',';      
        RecordReader testReader = new CSVRecordReader(numLinesToSkip, delimiter);        
        testReader.initialize(new FileSplit(new File(TestdatacsvPath)));
        DataSetIterator Testiterator = new RecordReaderDataSetIterator.Builder(testReader, batchSize)
     		   .classification(labelIndex, numClasses)        		
     	       .build();
        
        DataSet allData = Testiterator.next();
        allData.shuffle();

        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(allData.getFeatureMatrix());
        eval.eval(allData.getLabels(),output);
        log.info(eval.stats());
        
        //List<DataSet> testData = new ArrayList<>();
        //while(Testiterator.hasNext()) testData.add(Testiterator.next());  
		
		

	}

}
