package timeseriesweka.tcl;

import org.junit.Test;
import scala.tools.jline_embedded.internal.TestAccessible;
import timeseriesweka.classifiers.FastWWS.tools.UCR2CSV;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class CSV2ARFF {
    public static int numAtt=100;
    /**
     * Read train and test set from the given path
     * @param path
     * @param name
     * @return
     */
    @SuppressWarnings("all")
    public static Instances[] readTrainAndTest(String path, String name) {
        File trainFile = new File(path + name + "/" + name + "_TRAIN");
        if (!new File(trainFile.getAbsolutePath() + ".csv").exists()) {
            UCR2CSV.run(trainFile, new File(trainFile.getAbsolutePath() + ".csv"));
        }
        trainFile = new File(trainFile.getAbsolutePath() + ".csv");

        File testFile = new File(path + name + "/" + name + "_TEST");
        if (!new File(testFile.getAbsolutePath() + ".csv").exists()) {
            UCR2CSV.run(testFile, new File(testFile.getAbsolutePath() + ".csv"));
        }
        testFile = new File(testFile.getAbsolutePath() + ".csv");

        CSVLoader loader = new CSVLoader();
        Instances trainDataset = null;
        Instances testDataset = null;

        try {
            loader.setFile(trainFile);
            loader.setNominalAttributes(""+(numAtt+1));
            trainDataset = loader.getDataSet();
            trainDataset.setClassIndex(numAtt);

            loader.setFile(testFile);
//            loader.setNominalAttributes(""+(numAtt+1));
            testDataset = loader.getDataSet();
            testDataset.setClassIndex(numAtt);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return new Instances[] { trainDataset, testDataset };
    }
    /**
     * Read train and test set from the given path
     * @param path
     * @param name
     * @return
     */
    @SuppressWarnings("all")
    public static Instances[] readTrainAndTestTraffic(String path, String name) {
        File trainFile = new File(path + name + "/" + name + "_TRAIN");
        if (!new File(trainFile.getAbsolutePath() + ".csv").exists()) {
            UCR2CSV.run(trainFile, new File(trainFile.getAbsolutePath() + ".csv"));
        }
        trainFile = new File(trainFile.getAbsolutePath() + ".csv");

        File testFile = new File(path + name + "/" + name + "_TEST");
        if (!new File(testFile.getAbsolutePath() + ".csv").exists()) {
            UCR2CSV.run(testFile, new File(testFile.getAbsolutePath() + ".csv"));
        }
        testFile = new File(testFile.getAbsolutePath() + ".csv");

        CSVLoader loader = new CSVLoader();
        Instances trainDataset = null;
        Instances testDataset = null;

        try {
            loader.setFile(trainFile);
//            loader.setNominalAttributes(""+(numAtt+3));
//            loader.setNominalAttributes(""+(numAtt+4)); // posid，第605个属性
//            loader.setNominalAttributes(""+(numAtt+5)); // vehid，第 606个属性
            trainDataset = loader.getDataSet();
//            trainDataset.setClassIndex(numAtt+3); //gpsnum，第604个属性

            loader.setFile(testFile);
//            loader.setNominalAttributes(""+(numAtt+1));
//            loader.setNominalAttributes(""+(numAtt+4)); // posid
//            loader.setNominalAttributes(""+(numAtt+5));// vehid
            testDataset = loader.getDataSet();
//            testDataset.setClassIndex(numAtt+3);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return new Instances[] { trainDataset, testDataset };
    }

    @Test
    public void transferTheory(){
            String name = "theory";
            String path = "D:\\datasets\\Univariate_arff\\";
            Instances[] trainTest = readTrainAndTest(path,name);
            Instances train = trainTest[0];
            Instances test = trainTest[1];
            System.out.println(train);
//        for(int i =0;i<train.numInstances();i++){
//            Instance instance =train.instance(i);
//            double temp = instance.value(instance.numValues()-1);
//            instance.setValue(instance.numValues()-1,instance.value(0));
//            instance.setValue(0,temp);
//        }
//
//        for(int i =0;i<test.numInstances();i++){
//            Instance instance =test.instance(i);
//            double temp = instance.value(instance.numValues()-1);
//            instance.setValue(instance.numValues()-1,instance.value(0));
//            instance.setValue(0,temp);
//        }
            File trainFile = new File(path+name+"/"+name+"_TRAIN.arff");
            File testFile = new File(path+name+"/"+name+"_TEST.arff");
            try {
                BufferedWriter bw = new BufferedWriter(new FileWriter(trainFile));
                bw.write(train.toString());
                bw.close();
                bw = new BufferedWriter(new FileWriter(testFile));
                bw.write(test.toString());
                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
    }

    @Test
    /**
     * next: Efficient1NNTraffic
     */
    public void transferTraffic(){
        numAtt=600;
        String name = "Traffic";
        String path = "D:\\datasets\\Univariate_arff\\";
        Instances[] trainTest = readTrainAndTestTraffic(path,name);

        Instances train = trainTest[0];
        Instances test = trainTest[1];
        System.out.println(train);
        File trainFile = new File(path+name+"/"+name+"_TRAIN.arff");
        File testFile = new File(path+name+"/"+name+"_TEST.arff");
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(trainFile));
            bw.write(train.toString());
            bw.close();

            bw = new BufferedWriter(new FileWriter(testFile));
            bw.write(test.toString());
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String []args){

    }
}
