package timeseriesweka.classifiers.ensembles.elastic_ensemble;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.locks.ReentrantLock;

import javafx.util.Pair;
import timeseriesweka.filters.DerivativeFilter;
import utilities.ClassifierResults;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.classifiers.AbstractClassifier;
import development.ExperimentsClean;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * An abstract class to allow for distance-specific optimisations of DTW for use 
 * in the ElasticEnsemble. This class allows for univariate and multivariate
 * time series to be used; the multivariate version calculates distances as the
 * sum of individual distance calculations between common dimensions of two
 * instances (using the same parameter setting on all channels).
 * 
 * E.G. a DTW implementation with window = 0.5 (50%) for two instances with 10 
 * channels would calculate the DTW distance separately for each channel, and 
 * sum the 10 distances together. 
 * 
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public abstract class Efficient1NN extends AbstractClassifier implements SaveParameterInfo{
    
    protected Instances train;
    protected  DerivativeFilter derf;
    protected Instances[] trainGroup;
    protected String classifierIdentifier;
    protected boolean allowLoocv = true;
    protected int cparamId ;    // @Author tcl
    public double step_p;
    protected boolean singleParamCv = false; 
    
    private boolean fileWriting = false;
    private boolean individualCvParamFileWriting = false;
    private String outputDir;
    private String datasetName;
    private int resampleId;
    private ClassifierResults res =new ClassifierResults();
    public CheckPoint checkPoint = new CheckPoint();
    public int numThreads =4;
    public ExecutorService service = Executors.newFixedThreadPool(numThreads);
    public boolean saveCheckPoint = true;
    public boolean saveTrainTime= false;

    /**
     * Abstract method to calculates the distance between two Instance objects
     * @param first 
     * @param second
     * @param cutOffValue a best-so-far value to allow early abandons
     * @return the distance between first and second. If early abandon occurs, it will return Double.MAX_VALUE.
     */
    public abstract double distance(Instance first, Instance second, double cutOffValue);
    public  Pair<Double,double[][]> distanceSavePath(Instance first, Instance second, double cutoff){
        return new Pair<>(Double.MAX_VALUE,new double[1][1]);
    };
    /**
     * Multi-dimensional equivalent of the univariate distance method. Iterates 
     * through channels calculating distances independently using the same param
     * options, summing together at the end to return a single distance. 
     * 
     * @param first
     * @param second
     * @param cutOffValue
     * @return 
     */
    public double distance(Instance[] first, Instance[] second, double cutOffValue){
        double sum = 0;
        double decliningCutoff = cutOffValue;
        double thisDist;
        for(int d = 0; d < first.length; d++){
            thisDist = this.distance(first[d], second[d], decliningCutoff);
            sum += thisDist;
            if(sum > cutOffValue){
                return Double.MAX_VALUE;
            }
            decliningCutoff -= thisDist;
        }
        
        return sum;
    }
    
    /**
     * Utility method for easy cross-validation experiments. Each inheriting 
     * class has 100 param options to select from (some dependent on information
     * for the training data). Passing in the training data and a param
     * 
     * 
     * @param train
     * @param paramId 
     */
    public abstract void setParamsFromParamId(Instances train, int paramId);
    
    public void buildClassifier(Instances train) throws Exception{
        this.train = train;
        this.trainGroup = null;
    }
    
    public void buildClassifier(Instances[] trainGroup) throws Exception{
        this.train = null;
        this.trainGroup = trainGroup;
    }


    public double classifyInstanceOld(Instance instance) throws Exception {

        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.train.numClasses()];

        double thisDist;

        for(Instance i:this.train){
            thisDist = distance(instance, i, bsfDistance);
            if(thisDist < bsfDistance){
                bsfDistance = thisDist;
                classCounts = new int[train.numClasses()];
                classCounts[(int)i.classValue()]++;
            }else if(thisDist==bsfDistance){
                classCounts[(int)i.classValue()]++;
            }
        }

        double bsfClass = -1;
        double bsfCount = -1;
        for(int c = 0; c < classCounts.length; c++){
            if(classCounts[c]>bsfCount){
                bsfCount = classCounts[c];
                bsfClass = c;
            }
        }

        return bsfClass;
    }

    /**
     * @author tcl
     * can not get right accuracy
     * @param instance
     * @return
     * @throws Exception
     */
    public double classifyInstanceMP1Fail(Instance instance) throws Exception {

        class classifyAll{
            // for tie splitting
            int[] classCounts = new int[train.numClasses()];
            int j_lock = -1;
            double thisDist;
            double bsfDistance = Double.MAX_VALUE;
            double bsfClass = -1;
            Object lock1 =new Object();
            Object lock2 =new Object();

            public double getCls(){
//                System.out.println("[MultiProcess] Total case: " +train.numInstances());
                ArrayList<Thread> jobs =  new ArrayList<Thread>();
                for(int k = 0;k<7;k++){
                    Thread t = new Thread(new classifyOne(), String.valueOf(k));
                    t.start();
                    jobs.add(t);
                }
                for(Thread job: jobs){
                    try {
                        job.join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                double bsfCount = -1;
                for(int c = 0; c < classCounts.length; c++){
                    if(classCounts[c]>bsfCount){
                        bsfCount = classCounts[c];
                        bsfClass = c;
                    }
                }
                return bsfClass;
            }
            class classifyOne implements Runnable{

                @Override
                public void run() {
                    Instance i;
                    while(true){
                        int j = 0;
                        synchronized (lock1){
                            j_lock++;
                            j = j_lock;
                        }
                        if(j>train.numInstances()-1){
                            break;
                        }

                        if(j%1000==0)
                            System.out.println("[MultiProcess] Total case: " +train.numInstances()+" Test case: "+ j+" Thread: " +Thread.currentThread().getName());
                        i = train.get(j);
                        double thisDist = distance(instance, i, bsfDistance);
                        synchronized (lock2) {
                            if (thisDist < bsfDistance) {
                                bsfDistance = thisDist;
                                classCounts = new int[train.numClasses()];
                                classCounts[(int) i.classValue()]++;
                            } else if (thisDist == bsfDistance) {
                                classCounts[(int) i.classValue()]++;
                            }
                        }
                    }

                }
            }

        }
        return new classifyAll().getCls();
    }

    /**
     * @author tcl
     * can get right accuracy
     * @param instance
     * @return
     * @throws Exception
     */
    public double classifyInstanceMP(Instance instance) throws Exception {

        class classifyAll{
            // for tie splitting
            int[] classCounts = new int[train.numClasses()];
            int j = -1;
            double thisDist;
            double bsfDistance = Double.MAX_VALUE;
            double bsfClass = -1;
            Object lock1 =new Object();
            Object lock2 =new Object();

            public double getCls(){
//                System.out.println("[MultiProcess] Total case: " +train.numInstances());
                ArrayList<Thread> jobs =  new ArrayList<Thread>();
                for(int k = 0;k<3;k++){
                    Thread t = new Thread(new classifyOne(), String.valueOf(k));
                    t.start();
                    jobs.add(t);
                }
                for(Thread job: jobs){
                    try {
                        job.join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                double bsfCount = -1;
                for(int c = 0; c < classCounts.length; c++){
                    if(classCounts[c]>bsfCount){
                        bsfCount = classCounts[c];
                        bsfClass = c;
                    }
                }
                return bsfClass;
            }
            class classifyOne implements Runnable{

                @Override
                public void run() {
                    Instance i;
                    while(true){
                        synchronized (lock1){
                            if(j>train.numInstances()-2){
                                break;
                            }
                            j++;
                            if(j%100==0)
                                System.out.println("[MultiProcess] Total case: " +train.numInstances()+" Test case: "+ j+" Thread: " +Thread.currentThread().getName());
                            i = train.get(j);
                        }
                        double thisDist = distance(instance, i, bsfDistance);
                        synchronized (lock2) {
                            if (thisDist < bsfDistance) {
                                bsfDistance = thisDist;
                                classCounts = new int[train.numClasses()];
                                classCounts[(int) i.classValue()]++;
                            } else if (thisDist == bsfDistance) {
                                classCounts[(int) i.classValue()]++;
                            }
                        }
                    }

                }
            }

        }
        return new classifyAll().getCls();
    }

    /**
     * @author tcl
     * can get right accuracy
     * @param instance
     * @return
     * @throws Exception
     */
    public double classifyInstance(Instance instance) throws Exception {

        class classifyAll{
            // for tie splitting
            int[] classCounts = new int[train.numClasses()];
            int j = -1;
            double thisDist;
            double bsfDistance = Double.MAX_VALUE;
            double bsfClass = -1;
            Object lock1 =new Object();
            Object lock2 =new Object();

            public double getCls(){
//                System.out.println("[MultiProcess] Total case: " +train.numInstances());
                Collection<Future<?>> jobs =  new LinkedList<Future<?>>();
                classifyOne cone = new classifyOne();
                Future<?> future;
                for(int k = 0;k<numThreads;k++){
                    future = service.submit(cone);
                    jobs.add(future);
                }
                for(Future<?> job: jobs){
                    try {
                        job.get();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    } catch (ExecutionException e) {
                        e.printStackTrace();
                    }
                }
                double bsfCount = -1;
                for(int c = 0; c < classCounts.length; c++){
                    if(classCounts[c]>bsfCount){
                        bsfCount = classCounts[c];
                        bsfClass = c;
                    }
                }
                return bsfClass;
            }
            class classifyOne implements Runnable{

                @Override
                public void run() {
                    Instance i;
                    while(true){
                        synchronized (lock1){
                            if(j>train.numInstances()-2){
                                break;
                            }
                            j++;
                            if(j%100==0)
                                System.out.println("[MultiProcess] Total case: " +train.numInstances()+" Test case: "+ j+" Thread: " +Thread.currentThread().getName());
                            i = train.get(j);
                        }
                        double thisDist = distance(instance, i, bsfDistance);
                        synchronized (lock2) {
                            if (thisDist < bsfDistance) {
                                bsfDistance = thisDist;
                                classCounts = new int[train.numClasses()];
                                classCounts[(int) i.classValue()]++;
                            } else if (thisDist == bsfDistance) {
                                classCounts[(int) i.classValue()]++;
                            }
                        }
                    }

                }
            }

        }
        return new classifyAll().getCls();
    }

    public List<Pair> TraceBack(double[][] matrixD){
        List<Pair> path = new ArrayList<Pair>();
        int m = matrixD.length-1;
        int n = matrixD[0].length-1;
        while(n>0||m>0){
            path.add(new Pair(m,n));
            if(n>0&&m>0){
                double distMin = Math.min(matrixD[m-1][n-1],Math.min(matrixD[m][n-1],matrixD[m-1][n]));
                if(distMin==matrixD[m-1][n-1]){
                    m-=1;
                    n-=1;
                }else if(distMin==matrixD[m][n-1]){
                    n-=1;
                }else{
                    m-=1;
                }
            }else if(n>0){
                n-=1;
            }else{
                m-=1;
            }
        }
        path.add(new Pair(0,0));
        return path;
    }

    public List<Pair> TraceBackMDist(double[][] matrixD){
        /**
         * m is test dimension, n is the train dimension
         */
        List<Pair> path = new ArrayList<Pair>();
        int m = matrixD.length-1;
        int n = matrixD[0].length-1;
        double distMin;
        while(n>0||m>0){
            path.add(new Pair(m,matrixD[m][n]));
//            System.out.println(m+","+n+","+matrixD[m][n]);
            if(n>0&&m>0){
//                distMin = Math.min(matrixD[m-1][n-1],Math.min(matrixD[m][n-1],matrixD[n-1][m]));
                distMin = Math.min(matrixD[m-1][n-1],Math.min(matrixD[m][n-1],matrixD[m-1][n]));
                if(distMin==matrixD[m-1][n-1]){
                    m-=1;
                    n-=1;
                }else if(distMin==matrixD[m][n-1]){
                    n-=1;
                }else{
                    m-=1;
                }
            }else if(n>0){
                n-=1;
            }else{
                m-=1;
            }
        }
        path.add(new Pair(0,matrixD[0][0]));
        return path;
    }

    /**
     * @Author:tcl
     * @param instance
     * @return
     * @throws Exception
     */
    @SuppressWarnings("all")
    public Pair<Double, List> classifyInstanceSaveDistAndPath(Instance instance) throws Exception {

        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.train.numClasses()];

        double thisDist;
        Pair<Double,double[][]> distAndPaths ;
        List distPathList = new ArrayList<Pair>();
        for(int j=0;j<this.train.numInstances();j++){
            Instance i = this.train.get(j);
            distAndPaths = distanceSavePath(instance, i, bsfDistance); //instance is test
            thisDist = distAndPaths.getKey();
            distPathList.add(new Pair<>(thisDist,TraceBack(distAndPaths.getValue())));
//            distPathList.add(new Pair<>(instance.value(0),TraceBack(distAndPaths.getValue())));
            if(thisDist < bsfDistance){
                bsfDistance = thisDist;
                classCounts = new int[train.numClasses()];
                try{
                    classCounts[(int)i.classValue()]++;
                }catch (Exception e){
                    System.out.println(195+","+(int)i.classValue());
                    System.out.println(196+","+classCounts);
                }

            }else if(thisDist==bsfDistance){
                classCounts[(int)i.classValue()]++;
            }
        }

        double bsfClass = -1;
        double bsfCount = -1;
        for(int c = 0; c < classCounts.length; c++){
            if(classCounts[c]>bsfCount){
                bsfCount = classCounts[c];
                bsfClass = c;
            }
        }
        return new Pair<>(bsfClass,distPathList);
    }

    public Pair<Double, List> classifyInstanceSaveDistAndPathDist(Instance instance) throws Exception {

        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.train.numClasses()];

        double thisDist;
        Pair<Double,double[][]> distAndPaths ;
        List distPathList = new ArrayList<Pair>();
        for(int j=0;j<this.train.numInstances();j++){
            Instance i = this.train.get(j);
            distAndPaths = distanceSavePath(instance, i, bsfDistance); //instance is test
            thisDist = distAndPaths.getKey();
            distPathList.add(new Pair<>(thisDist,TraceBackMDist(distAndPaths.getValue())));
//            distPathList.add(new Pair<>(instance.value(0),TraceBack(distAndPaths.getValue())));
            if(thisDist < bsfDistance){
                bsfDistance = thisDist;
                classCounts = new int[train.numClasses()];
                try{
                    classCounts[(int)i.classValue()]++;
                }catch (Exception e){
                    System.out.println(195+","+(int)i.classValue());
                    System.out.println(196+","+classCounts);
                }

            }else if(thisDist==bsfDistance){
                classCounts[(int)i.classValue()]++;
            }
        }

        double bsfClass = -1;
        double bsfCount = -1;
        for(int c = 0; c < classCounts.length; c++){
            if(classCounts[c]>bsfCount){
                bsfCount = classCounts[c];
                bsfClass = c;
            }
        }
        return new Pair<>(bsfClass,distPathList);
    }

    public double classifyInstanceMultivariate(Instance[] instance) throws Exception {
    
        if(this.trainGroup==null){
            throw new Exception("Error: this configuration is for multivariate data");
        }
        
        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.trainGroup[0].numClasses()];
        
        double thisDist;
            
        Instance[] trainInstancesByDimension;
        for(int i = 0; i < this.trainGroup[0].numInstances(); i++){
            trainInstancesByDimension = new Instance[this.trainGroup.length];
            for(int j = 0; j < trainInstancesByDimension.length; j++){
                trainInstancesByDimension[j] = this.trainGroup[j].instance(i);
            }
            
            thisDist = distance(instance, trainInstancesByDimension, bsfDistance);
            if(thisDist < bsfDistance){
                bsfDistance = thisDist;
                classCounts = new int[trainGroup[0].numClasses()];
                classCounts[(int)trainGroup[0].instance(i).classValue()]++;
            }else if(thisDist==bsfDistance){
                classCounts[(int)trainGroup[0].instance(i).classValue()]++;
            }
        }        
       
        double bsfClass = -1;
        double bsfCount = -1;
        for(int c = 0; c < classCounts.length; c++){
            if(classCounts[c]>bsfCount){
                bsfCount = classCounts[c];
                bsfClass = c;
            }
        }
        
        return bsfClass;
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        
        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.train.numClasses()];
        
        double thisDist;
        int sumOfBest = 0;
                
        for(Instance i:this.train){
            thisDist = distance(instance, i, bsfDistance); 
            if(thisDist < bsfDistance){
                bsfDistance = thisDist;
                classCounts = new int[train.numClasses()];
                classCounts[(int)i.classValue()]++;
                sumOfBest = 1;
            }else if(thisDist==bsfDistance){
                classCounts[(int)i.classValue()]++;
                sumOfBest++;
            }
        }
        
        double[] classDistributions = new double[this.train.numClasses()];
        for(int c = 0; c < classCounts.length; c++){
            classDistributions[c] = (double)classCounts[c]/sumOfBest;
        }
 
        return classDistributions;
    }
    
    public void setClassifierIdentifier(String classifierIdentifier){
        this.classifierIdentifier = classifierIdentifier;
    }
    
    public String getClassifierIdentifier(){
        return classifierIdentifier;
    } 
    
    @Override
    public String getParameters(){
        String paras="BuildTime,"+res.buildTime;
        return paras;
        
    }    
    // could parallelise here
//    public void writeLOOCVOutput(String tscProblemDir, String datasetName, int resampleId, String outputResultsDir, boolean tidyUp) throws Exception{    
//        for(int paramId = 0; paramId < 100; paramId++){
//            writeLOOCVOutput(tscProblemDir, datasetName, resampleId, outputResultsDir, paramId);
//        }    
//        parseLOOCVResults(tscProblemDir, datasetName, resampleId, outputResultsDir, tidyUp);
//    }
//    
//    public double writeLOOCVOutput(String tscProblemDir, String datasetName, int resampleId, String outputResultsDir, int paramId) throws Exception{
//        new File(outputResultsDir+classifierIdentifier+"/Predictions/"+datasetName+"/loocvForParamOptions/").mkdirs();
//        
//        Instances train = ClassifierTools.loadData(tscProblemDir+datasetName+"/"+datasetName+"_TRAIN");
//        Instances test = ClassifierTools.loadData(tscProblemDir+datasetName+"/"+datasetName+"_TEST");
//        
//        if(resampleId!=0){
//            Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, resampleId);
//            train = temp[0];
//            test = temp[1];
//        }
//        
//        this.setParamsFromParamId(paramId);
//        
//        Instances trainLoocv;
//        Instance testLoocv;
//        
//        int correct = 0;
//        double pred, actual;
//        for(int i = 0; i < train.numInstances(); i++){
//            trainLoocv = new Instances(train);
//            testLoocv = trainLoocv.remove(i);
//            actual = testLoocv.classValue();
//            this.buildClassifier(train);
//            pred = this.classifyInstance(testLoocv);
//            if(pred==actual){
//                correct++;
//            }
//        }
//        
//        return (double)correct/train.numInstances();
//    }
    
    
    public void setFileWritingOn(String outputDir, String datasetName, int resampleId){
        this.fileWriting = true;
        this.outputDir = outputDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
    }
    public void setIndividualCvFileWritingOn(String outputDir, String datasetName, int resampleId){
        this.individualCvParamFileWriting = true;
        this.outputDir = outputDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
    }

    @SuppressWarnings("all")
    public double[] loocvStep(Instances train,Instances test) throws Exception{
        double[] accAndPreds = null;
        String parsedFileName = this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv";

//        System.out.println(parsedFileName);

        if(fileWriting){
            File existing = new File(parsedFileName);
            if(existing.exists()){
//                throw new Exception("Parsed results already exist for this measure: "+ parsedFileName);
                Scanner scan = new Scanner(existing);
                scan.useDelimiter("\n");
                scan.next(); // skip header line
                int paramId = Integer.parseInt(scan.next().trim().split(",")[0]);
                if(this.allowLoocv){
                    this.setParamsFromParamId(train, paramId);
                }
                this.buildClassifier(train);
                accAndPreds = new double[train.numInstances()+1];
                accAndPreds[0] = Double.parseDouble(scan.next().trim().split(",")[0]);
                int correct = 0;
                String[] temp;
                for(int i = 0; i < train.numInstances(); i++){
                    temp = scan.next().split(",");
                    accAndPreds[i+1] = Double.parseDouble(temp[1]);
                    if(accAndPreds[i+1]==Double.parseDouble(temp[0])){
                        correct++;
                    }
                }
                // commented out for now as this breaks the new EE loocv thing we're doing for the competition. Basically, if we try and load for train-1 ins for test in loocv, the number of train instances doesn't match so the acc is slightly off. should be an edge case, but can leave this check out so long as we trust the code
//                if(((double)correct/train.numInstances())!=accAndPreds[0]){
//                    System.err.println(existing.getAbsolutePath());
//                    System.err.println(((double)correct/train.numInstances())+" "+accAndPreds[0]);
//                    throw new Exception("Attempted file loading, but accuracy doesn't match itself?!");
//                }
                return accAndPreds;
            }else{
                new File(this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/").mkdirs();
            }
        }

//        write output
//        maybe a different version which looks for missing files and runs them?


        double bsfAcc = -1;
        int bsfParamId = -1;
        double[] bsfaccAndPreds = null;

        int paramIdTmp = this.cparamId>0? this.cparamId:100; //tcl
        String[] totalClassName =this.getClass().toString().split(" ")[1].split("[.]");
        String stepPathDir = "D:/硕士培养手册/小论文/results/step/"+totalClassName[totalClassName.length-1]+"/"+datasetName;
        File stepAcc = new File(stepPathDir+"/stepAcc.txt");
        if(!stepAcc.getParentFile().exists()){
            stepAcc.getParentFile().mkdirs();
        }
        BufferedWriter bw = new BufferedWriter(new FileWriter(stepAcc,false));
        StringBuffer stepacc = new StringBuffer("");
        StringBuffer stepaccTest = new StringBuffer("");
        double[] accAndPredsTest = null;
        for(int paramId = 0; paramId < paramIdTmp; paramId++){
//            System.out.print(paramId+" ");
            accAndPreds = loocvAccAndPreds(train,paramId);
            accAndPredsTest = loocvAccAndPredsTest(train,test,paramId);
//            System.out.println(this.allowLoocv);
//            System.out.println(accAndPreds[0]);
//            System.out.println(paramId+","+this.getParamInformationString()+","+accAndPreds[0]);
            stepacc.append(paramId+","+accAndPreds[0]+";");
            stepaccTest.append(paramId+","+accAndPredsTest[0]+";");
            if(accAndPreds[0]>bsfAcc){
                bsfAcc = accAndPreds[0];
                bsfParamId = paramId;
                bsfaccAndPreds = accAndPreds;
            }
            if(!this.allowLoocv){
                paramId = paramIdTmp;
            }
        }
//        System.out.println(this.classifierIdentifier+", bsfParamId "+bsfParamId);
        this.buildClassifier(train);
        bw.write(stepacc.toString()+"\n");
        bw.write(stepaccTest.toString());
        bw.flush();
        bw.close();
        if(this.allowLoocv){
            this.setParamsFromParamId(train, bsfParamId);
        }
        if(fileWriting){
            FileWriter out = new FileWriter(parsedFileName);
            out.append(this.classifierIdentifier+","+datasetName+",parsedTrain\n");
            out.append(bsfParamId+"\n");
            out.append(bsfAcc+"\n");
            for(int i = 1; i < bsfaccAndPreds.length; i++){
                out.append(train.instance(i-1).classValue()+","+bsfaccAndPreds[i]+"\n");
            }
            out.close();
        }

        return bsfaccAndPreds;
    }

    public double[] loocv(Instances train) throws Exception{
        double[] accAndPreds = null;
        String parsedFileName = this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv";
        
//        System.out.println(parsedFileName);
        
        if(fileWriting){
            File existing = new File(parsedFileName);
            if(existing.exists()){
//                throw new Exception("Parsed results already exist for this measure: "+ parsedFileName);
                Scanner scan = new Scanner(existing);
                scan.useDelimiter("\n");
                scan.next(); // skip header line
                int paramId = Integer.parseInt(scan.next().trim().split(",")[0]);
                if(this.allowLoocv){
                    this.setParamsFromParamId(train, paramId);
                }
                this.buildClassifier(train);
                accAndPreds = new double[train.numInstances()+1];
                accAndPreds[0] = Double.parseDouble(scan.next().trim().split(",")[0]);
                int correct = 0;
                String[] temp;
                for(int i = 0; i < train.numInstances(); i++){
                    temp = scan.next().split(",");
                    accAndPreds[i+1] = Double.parseDouble(temp[1]);
                    if(accAndPreds[i+1]==Double.parseDouble(temp[0])){
                        correct++;
                    }
                }
                // commented out for now as this breaks the new EE loocv thing we're doing for the competition. Basically, if we try and load for train-1 ins for test in loocv, the number of train instances doesn't match so the acc is slightly off. should be an edge case, but can leave this check out so long as we trust the code
//                if(((double)correct/train.numInstances())!=accAndPreds[0]){
//                    System.err.println(existing.getAbsolutePath());
//                    System.err.println(((double)correct/train.numInstances())+" "+accAndPreds[0]);
//                    throw new Exception("Attempted file loading, but accuracy doesn't match itself?!");
//                }
                return accAndPreds;
            }else{
                new File(this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/").mkdirs();
            }
        }
        
//        write output
//        maybe a different version which looks for missing files and runs them?
        

        double bsfAcc = -1;
        int bsfParamId = -1;
        double[] bsfaccAndPreds = null;

        int paramIdTmp = this.cparamId>0? this.cparamId:100; //tcl
        String[] totalClassName =this.getClass().toString().split(" ")[1].split("[.]");
        String stepPathDir = "D:/硕士培养手册/小论文/results/step/"+totalClassName[totalClassName.length-1]+"/"+datasetName;
        File stepAcc = new File(stepPathDir+"/stepAcc.txt");
        if(!stepAcc.getParentFile().exists()){
            stepAcc.getParentFile().mkdirs();
        }
        BufferedWriter bw = new BufferedWriter(new FileWriter(stepAcc,false));
        StringBuffer stepacc = new StringBuffer("");
        for(int paramId = 0; paramId < paramIdTmp; paramId++){
//            System.out.print(paramId+" ");

            // jump to validateId
            if(paramId<checkPoint.getValidateId()){
                paramId = checkPoint.getValidateId();
                System.out.println("[checkpoint] starting from paramid "+paramId);
                continue;
            }
            accAndPreds = loocvAccAndPreds(train,paramId);
//            System.out.println(this.allowLoocv);
//            System.out.println(accAndPreds[0]);
//            System.out.println(paramId+","+this.getParamInformationString()+","+accAndPreds[0]);
            stepacc.append(bsfParamId+","+accAndPreds[0]+";");
            if(accAndPreds[0]>bsfAcc){
                bsfAcc = accAndPreds[0];
                bsfParamId = paramId;
                checkPoint.setValidateBest(new Pair<>(bsfParamId,bsfAcc));
                bsfaccAndPreds = accAndPreds;
            }

            // save for every validate
            checkPoint.setValidateId(paramId);
            checkPoint.saveParas();
            if(!this.allowLoocv){
                paramId = paramIdTmp;
            }
        }
//        System.out.println(this.classifierIdentifier+", bsfParamId "+bsfParamId);
        this.buildClassifier(train);
        bw.write(stepacc.toString());
        bw.flush();
        bw.close();
        if(this.allowLoocv){
            bsfParamId = checkPoint.getValidateBest().getKey();
            this.setParamsFromParamId(train, bsfParamId);
        }    
        if(fileWriting){
            FileWriter out = new FileWriter(parsedFileName);
            out.append(this.classifierIdentifier+","+datasetName+",parsedTrain\n");
            out.append(bsfParamId+"\n");
            out.append(bsfAcc+"\n");
            for(int i = 1; i < bsfaccAndPreds.length; i++){
                out.append(train.instance(i-1).classValue()+","+bsfaccAndPreds[i]+"\n");
            }
            out.close();
        }
        
        return bsfaccAndPreds;
    }
    
    DecimalFormat df = new DecimalFormat("##.###");

    @Deprecated
    public double[] loocv(Instances[] trainGroup) throws Exception{
        double[] accAndPreds = null;
        String parsedFileName = this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv";
        
        Instances concatenated = concatenate(trainGroup);
        
        
        if(fileWriting){
            File existing = new File(parsedFileName);
            if(existing.exists()){
//                throw new Exception("Parsed results already exist for this measure: "+ parsedFileName);
                Scanner scan = new Scanner(existing);
                scan.useDelimiter("\n");
                scan.next(); // skip header line
                int paramId = Integer.parseInt(scan.next().trim().split(",")[0]);
                if(this.allowLoocv){
                    this.setParamsFromParamId(concatenated, paramId);
                }
                this.buildClassifier(trainGroup);
                accAndPreds = new double[trainGroup[0].numInstances()+1];
                accAndPreds[0] = Double.parseDouble(scan.next().trim().split(",")[0]);
                int correct = 0;
                String[] temp;
                for(int i = 0; i < trainGroup[0].numInstances(); i++){
                    temp = scan.next().split(",");
                    accAndPreds[i+1] = Double.parseDouble(temp[1]);
                    if(accAndPreds[i+1]==Double.parseDouble(temp[0])){
                        correct++;
                    }
                }
                // commented out for now as this breaks the new EE loocv thing we're doing for the competition. Basically, if we try and load for train-1 ins for test in loocv, the number of train instances doesn't match so the acc is slightly off. should be an edge case, but can leave this check out so long as we trust the code
//                if(((double)correct/train.numInstances())!=accAndPreds[0]){
//                    System.err.println(existing.getAbsolutePath());
//                    System.err.println(((double)correct/train.numInstances())+" "+accAndPreds[0]);
//                    throw new Exception("Attempted file loading, but accuracy doesn't match itself?!");
//                }
                return accAndPreds;
            }else{
                new File(this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/").mkdirs();
            }
        }
        
//        write output
//        maybe a different version which looks for missing files and runs them?
        

        double bsfAcc = -1;
        int bsfParamId = -1;
        double[] bsfaccAndPreds = null;

        for(int paramId = 0; paramId < 100; paramId++){
//            System.out.print(paramId+" ");
            accAndPreds = loocvAccAndPreds(trainGroup,concatenated,paramId);
//            System.out.println(this.allowLoocv);
//            System.out.println(accAndPreds[0]);
            if(accAndPreds[0]>bsfAcc){
                bsfAcc = accAndPreds[0];
                bsfParamId = paramId;
                bsfaccAndPreds = accAndPreds;
            }
            System.out.println("\t"+paramId+": "+df.format(accAndPreds[0]*100)+" ("+df.format(bsfAcc*100)+")");
            if(!this.allowLoocv){
                paramId = 100;
            }
        }
//        System.out.println(this.classifierIdentifier+", bsfParamId "+bsfParamId);
        this.buildClassifier(trainGroup);
        if(this.allowLoocv){
            this.setParamsFromParamId(concatenated, bsfParamId);
        }    
        if(fileWriting){
            FileWriter out = new FileWriter(parsedFileName);
            out.append(this.classifierIdentifier+","+datasetName+",parsedTrain\n");
            out.append(bsfParamId+"\n");
            out.append(bsfAcc+"\n");
            for(int i = 1; i < bsfaccAndPreds.length; i++){
                out.append(trainGroup[0].instance(i-1).classValue()+","+bsfaccAndPreds[i]+"\n");
            }
            out.close();
        }
        
        return bsfaccAndPreds;
    }
    
    public double[] loocvAccAndPreds(Instances train, int paramId) throws Exception{
        if(this.allowLoocv){
            this.setParamsFromParamId(train, paramId);
        }
        FileWriter out = null;
        String parsedFileName = this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv";
        String singleFileName = this.outputDir+this.classifierIdentifier+"/cv/"+datasetName+"/trainFold"+resampleId+"/pid"+paramId+".csv";
        if(this.individualCvParamFileWriting){
            if(new File(parsedFileName).exists()){//|| new File(singleFileName).exists()){
                throw new Exception("Error: Full parsed training results already exist - "+parsedFileName);
            }else if(new File(singleFileName).exists()){
                throw new Exception("Error: CV training results already exist for this pid - "+singleFileName);
            }
        }
        
        
        // else we already know what the params are, so don't need to set
        
        Instances trainLoocv;
        Instance testLoocv;
        
        int correct = 0;
        double pred, actual;
        
        double[] accAndPreds = new double[train.numInstances()+1];
        for(int i = 0; i < train.numInstances(); i++){
            if(i%10==0)
                System.out.println("[Loop] Validate train case: " + i);
            if(i<checkPoint.getValidateNum()){
                i = checkPoint.getValidateNum();
                correct =checkPoint.getValidateCorrect();
                System.out.println("[checkpoint] validate id: "+checkPoint.getValidateId()+
                        " validateNum: "+ checkPoint.getValidateNum()+"validateCorrect: "+checkPoint.getValidateCorrect());
                continue;
            }
            trainLoocv = new Instances(train);
            testLoocv = trainLoocv.remove(i);
            actual = testLoocv.classValue();
            this.buildClassifier(trainLoocv);
            pred = this.classifyInstance(testLoocv);
            if(pred==actual){
                correct++;
            }
            accAndPreds[i+1]= pred;
            if(i%10==0){
                checkPoint.setValidateNum(i);
                checkPoint.setValidateCorrect(correct);
                checkPoint.saveParas();
            }
        }
        accAndPreds[0] = (double)correct/train.numInstances();
//        System.out.println(accAndPreds[0]);
        
        if(individualCvParamFileWriting){
            new File(this.outputDir+this.classifierIdentifier+"/cv/"+datasetName+"/trainFold"+resampleId+"/").mkdirs();
            out = new FileWriter(singleFileName);
            out.append(this.classifierIdentifier+","+datasetName+",cv\n");
            out.append(paramId+"\n");
            out.append(accAndPreds[0]+"\n");
            for(int i = 1; i < accAndPreds.length;i++){
                out.append(train.instance(i-1).classValue()+","+accAndPreds[i]+"\n");
            }
            out.close();
        }

        checkPoint.setValidateNum(-1);
        checkPoint.setValidateCorrect(-1);
        checkPoint.saveParas();
        return accAndPreds;
    }

    /**
     * multiprocess at test case level
     * can run correctly, may fall into dead lock
     * and this is not synchronized  and even if this is synchronized , the efficiency is low
     * @param train
     * @param paramId
     * @return
     * @throws Exception
     */
    public double[] loocvAccAndPredsMP(Instances train, int paramId) throws Exception{
        if(this.allowLoocv){
            this.setParamsFromParamId(train, paramId);
        }
        FileWriter out = null;
        String parsedFileName = this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv";
        String singleFileName = this.outputDir+this.classifierIdentifier+"/cv/"+datasetName+"/trainFold"+resampleId+"/pid"+paramId+".csv";
        if(this.individualCvParamFileWriting){
            if(new File(parsedFileName).exists()){//|| new File(singleFileName).exists()){
                throw new Exception("Error: Full parsed training results already exist - "+parsedFileName);
            }else if(new File(singleFileName).exists()){
                throw new Exception("Error: CV training results already exist for this pid - "+singleFileName);
            }
        }


        // else we already know what the params are, so don't need to set
        Efficient1NN outoutTemp = this;

        class Out {

            double[] accAndPreds = new double[train.numInstances() + 1];
            int correct = 0;

            public Pair clsAll() {
                MultiProcess mp = new MultiProcess();
                List<Thread> jobs = new ArrayList<Thread>();
                int num = 7;
                for(int j = 0; j<num; j++){
                    jobs.add(new Thread(mp,String.valueOf(j)));
                }
                for(int j = 0; j<num; j++){
                    jobs.get(j).start();
                }
                for(int j = 0; j<num; j++){
                    try {
                        jobs.get(j).join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                return new Pair(accAndPreds,correct);
            }

            class MultiProcess implements Runnable{
                Efficient1NN outout = outoutTemp;

                int i_lock = 0;
                ReentrantLock lock1 = new ReentrantLock();
                ReentrantLock lock2 = new ReentrantLock();

                @Override
                public void run() {
                    while(true) {
                        Instances trainLoocv;
                        Instance testLoocv;
                        double pred, actual;

                        lock1.lock();
                        if(i_lock >= train.numInstances()){
                            break;
                        }
                        int i = i_lock;
                        if (i_lock < checkPoint.getValidateNum()) {
                            i = checkPoint.getValidateNum();
                            correct = checkPoint.getValidateCorrect();
                            System.out.println("[checkpoint] validate id: " + checkPoint.getValidateId() +
                                    " validateNum: " + checkPoint.getValidateNum() + "validateCorrect: " + checkPoint.getValidateCorrect());
                            continue;
                        }
                        System.out.println("[MultiProcess] case: " + i+
                                " Thread: "+Thread.currentThread().getName());
                        i_lock++;
                        lock1.unlock();



                        if (i % 10 == 0)
                            System.out.println("[Loop] Validate train case: " + i);

                        trainLoocv = new Instances(train);
                        testLoocv = trainLoocv.remove(i);



                        actual = testLoocv.classValue();
                        try{
                            outout.buildClassifier(trainLoocv);
                            pred = outout.classifyInstance(testLoocv);

                            lock2.lock();
                            if (pred == actual) {
                                correct++;
                            }
                            accAndPreds[i + 1] = pred;
                            lock2.unlock();

                        }catch (Exception e){
                            e.printStackTrace();
                        }

                        if (i % 10 == 0) {
                            checkPoint.setValidateNum(i);
                            checkPoint.setValidateCorrect(correct);
                            checkPoint.saveParas();
                        }
                    }
                }
            }


        }


//        double[] accAndPreds = new double[train.numInstances()+1];
        Pair pair =new Out().clsAll();
        double[] accAndPreds =(double[]) pair.getKey();
        accAndPreds[0] = (double)(int)pair.getValue()/train.numInstances();
//        System.out.println(accAndPreds[0]);

        if(individualCvParamFileWriting){
            new File(this.outputDir+this.classifierIdentifier+"/cv/"+datasetName+"/trainFold"+resampleId+"/").mkdirs();
            out = new FileWriter(singleFileName);
            out.append(this.classifierIdentifier+","+datasetName+",cv\n");
            out.append(paramId+"\n");
            out.append(accAndPreds[0]+"\n");
            for(int i = 1; i < accAndPreds.length;i++){
                out.append(train.instance(i-1).classValue()+","+accAndPreds[i]+"\n");
            }
            out.close();
        }

        checkPoint.setValidateNum(-1);
        checkPoint.setValidateCorrect(-1);
        checkPoint.saveParas();
        return accAndPreds;
    }
    @SuppressWarnings("all")
    public double[] loocvAccAndPredsTest(Instances train,Instances test, int paramId) throws Exception{
        if(this.allowLoocv){
            this.setParamsFromParamId(train, paramId);
        }
        FileWriter out = null;
        String parsedFileName = this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv";
        String singleFileName = this.outputDir+this.classifierIdentifier+"/cv/"+datasetName+"/trainFold"+resampleId+"/pid"+paramId+".csv";
        if(this.individualCvParamFileWriting){
            if(new File(parsedFileName).exists()){//|| new File(singleFileName).exists()){
                throw new Exception("Error: Full parsed training results already exist - "+parsedFileName);
            }else if(new File(singleFileName).exists()){
                throw new Exception("Error: CV training results already exist for this pid - "+singleFileName);
            }
        }


        // else we already know what the params are, so don't need to set

        Instances trainLoocv;
        Instance testLoocv;

        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[test.numInstances()+1];
        for(int i = 0; i < test.numInstances(); i++){
            trainLoocv = new Instances(train);
            testLoocv = test.instance(i);
            actual = testLoocv.classValue();
            this.buildClassifier(trainLoocv);
            pred = this.classifyInstance(testLoocv);
            if(pred==actual){
                correct++;
            }
            accAndPreds[i+1]= pred;
        }
        accAndPreds[0] = (double)correct/test.numInstances();
//        System.out.println(accAndPreds[0]);

        if(individualCvParamFileWriting){
            new File(this.outputDir+this.classifierIdentifier+"/cv/"+datasetName+"/trainFold"+resampleId+"/").mkdirs();
            out = new FileWriter(singleFileName);
            out.append(this.classifierIdentifier+","+datasetName+",cv\n");
            out.append(paramId+"\n");
            out.append(accAndPreds[0]+"\n");
            for(int i = 1; i < accAndPreds.length;i++){
                out.append(train.instance(i-1).classValue()+","+accAndPreds[i]+"\n");
            }
            out.close();
        }

        return accAndPreds;
    }

    public double[] loocvAccAndPreds(Instances[] trainGroup, Instances concatenated, int paramId) throws Exception{
        
        FileWriter out = null;
        String parsedFileName = this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv";
        String singleFileName = this.outputDir+this.classifierIdentifier+"/cv/"+datasetName+"/trainFold"+resampleId+"/pid"+paramId+".csv";
//        if(this.fileWriting){
//            if(new File(parsedFileName).exists()){//|| new File(singleFileName).exists()){
//                throw new Exception("Error: Full parsed training results already exist - "+parsedFileName);
//            }else if(new File(singleFileName).exists()){
//                throw new Exception("Error: CV training results already exist for this pid - "+singleFileName);
//            }
//        }
        
        if(this.allowLoocv){
//            System.out.println("allowed");
            this.setParamsFromParamId(concatenated, paramId);
//            System.out.println(this.toString());
        }
        // else we already know what the params are, so don't need to set
        
        Instances[] trainLoocv;
        Instance[] testLoocv;
        
        int correct = 0;
        double pred, actual;
        
        double[] accAndPreds = new double[trainGroup[0].numInstances()+1];
        for(int i = 0; i < trainGroup[0].numInstances(); i++){
            trainLoocv = new Instances[trainGroup.length];
            testLoocv = new Instance[trainGroup.length];
            
            for(int d = 0; d < trainGroup.length; d++){
                trainLoocv[d] = new Instances(trainGroup[d]);
                testLoocv[d] = trainLoocv[d].remove(i);
            }
            
//            trainLoocv = new Instances(train);
//            testLoocv = trainLoocv.remove(i);
            actual = testLoocv[0].classValue();
            this.buildClassifier(trainLoocv);
            pred = this.classifyInstanceMultivariate(testLoocv);

            if(pred==actual){
                correct++;
            }
            accAndPreds[i+1]= pred;
        }
        accAndPreds[0] = (double)correct/trainGroup[0].numInstances();
//        System.out.println(accAndPreds[0]);
        
//        if(fileWriting){
//            out = new FileWriter(singleFileName);
//            out.append(this.classifierIdentifier+","+datasetName+",cv\n");
//            out.append(paramId+"\n");
//            out.append(accAndPreds[0]+"\n");
//            for(int i = 1; i < accAndPreds.length;i++){
//                out.append(train.instance(i-1).classValue()+","+accAndPreds[i]+"\n");
//            }
//            out.close();
//        }
        
        return accAndPreds;
    }
    
    public void writeTrainTestOutput(String tscProblemDir, String datasetName, int resampleId, String outputResultsDir) throws Exception{
        
        // load in param id from training results
        File cvResults = new File(outputResultsDir+classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv");
        if(!cvResults.exists()){
            throw new Exception("Error loading file "+cvResults.getAbsolutePath());
        }
        Scanner scan = new Scanner(cvResults);
        scan.useDelimiter(System.lineSeparator());
        scan.next();
        int paramId = Integer.parseInt(scan.next().trim());
        this.setParamsFromParamId(train, paramId);
        
        // Now classifier is set up, make the associated files and do the test classification
        
        new File(outputResultsDir+classifierIdentifier+"/Predictions/"+datasetName+"/").mkdirs();
        StringBuilder headerInfo = new StringBuilder();
        
        headerInfo.append(classifierIdentifier).append(System.lineSeparator());
        headerInfo.append(this.getParamInformationString()).append(System.lineSeparator());
        
        Instances train = ClassifierTools.loadData(tscProblemDir+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData(tscProblemDir+datasetName+"/"+datasetName+"_TEST");
        
        if(resampleId!=0){
            Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, resampleId);
            train = temp[0];
            test = temp[1];
        }
        
        this.buildClassifier(train);
        StringBuilder classificationInfo = new StringBuilder();
        int correct = 0;
        double pred, actual;
        for(int i = 0; i < test.numInstances(); i++){
            actual = test.instance(i).classValue();
            pred = this.classifyInstance(test.instance(i));
            classificationInfo.append(actual).append(",").append(pred).append(System.lineSeparator());
            if(actual==pred){
                correct++;
            }
        }
        
        FileWriter outWriter = new FileWriter(outputResultsDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/testFold"+resampleId+".csv");
        outWriter.append(headerInfo);
        outWriter.append(((double)correct/test.numInstances())+System.lineSeparator());
        outWriter.append(classificationInfo);
        outWriter.close();
        
        
    }
    
//    public static void parseLOOCVResults(String tscProblemDir, String datasetName, int resampleId, String outputResultsDir, boolean tidyUp){
//        
//    }

    public abstract String getParamInformationString();
    
    public Instances getTrainingData(){  
        return this.train;
    }
    
    public static Instances concatenate(Instances[] train){
    // make a super arff for finding params that need stdev etc
        Instances temp = new Instances(train[0],0);
        for(int i = 1; i < train.length; i++){
            for(int j = 0; j < train[i].numAttributes()-1;j++){
                temp.insertAttributeAt(train[i].attribute(j), temp.numAttributes()-1);
            }
        }

        int dataset, attFromData;

        for(int insId = 0; insId < train[0].numInstances(); insId++){
            DenseInstance dense = new DenseInstance(temp.numAttributes());
            for(int attId = 0; attId < temp.numAttributes()-1; attId++){
            
                dataset = attId/(train[0].numAttributes()-1);
                attFromData = attId%(train[0].numAttributes()-1);
                dense.setValue(attId,train[dataset].instance(insId).value(attFromData));
                
            }
            dense.setValue(temp.numAttributes()-1, train[0].instance(insId).classValue());
            temp.add(dense);
        }
        return temp;
    }

    /**
     * @Author：tcl
     * Combination with children class function
     *      public static void ucr2018(){}
     */
    public void UCR2018Cls(boolean cv){
        UCR2018Cls(cv,false,"None");
    }
    /**
     * @Author：tcl
     * Combination with children class function
     *      public static void ucr2018(){}
     */
    public void UCR2018Cls(boolean cv,boolean resample100){
        UCR2018Cls(cv,resample100,"None");
    }
    public void UCR2018ClsSavePathAll(boolean cv, boolean resample100){
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(new File("D:\\硕士培养手册\\小论文\\dataForJavaCls.txt")));
            String datasetName;
            while((datasetName=br.readLine())!=null){
                UCR2018Cls(cv, resample100,datasetName,false);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    /**
     * @Author：tcl
     * Combination with children class function
     *      public static void ucr2018(){}
     */
    public void UCR2018Cls(boolean cv, boolean resample100,String datasetSavePath){
        this.UCR2018Cls(cv, resample100,datasetSavePath,false);
    }
    /**
     * @Author：tcl
     * Combination with children class function
     *      public static void ucr2018(){}
     */
    public void UCR2018Cls(boolean cv, boolean resample100,String datasetSavePath,boolean stepAnalyze){
        String[] totalClassName =this.getClass().toString().split(" ")[1].split("[.]");
        System.out.println(this.getClass().toString().split(" ")[1]+totalClassName.length);
        System.out.println("UCR_2018_"+totalClassName[totalClassName.length-1]+".txt");
        File resCls = new File("UCR_2018_"+totalClassName[totalClassName.length-1]+".txt");//
        if(resample100){
            resCls = new File("UCR_2018_Resample100_"+totalClassName[totalClassName.length-1]+".txt");
        }
        String distPathDir = "D:/硕士培养手册/小论文/results/distPath/"+totalClassName[totalClassName.length-1]+"/"+datasetSavePath;
        if(!datasetSavePath.equals("None")){
            resCls = new File("D:/硕士培养手册/小论文/results/distPath/"+totalClassName[totalClassName.length-1]+"/"+datasetSavePath+"/timestamp.txt");
            if(!resCls.getParentFile().exists()){
                resCls.getParentFile().mkdirs();
            }
            resCls = new File("D:/硕士培养手册/小论文/results/distPath/"+totalClassName[totalClassName.length-1]+"/datasetDone.txt");
            if(stepAnalyze) {
                resCls = new File("D:/硕士培养手册/小论文/results/step/" + totalClassName[totalClassName.length - 1] + "/datasetDone.txt");
                if(!resCls.exists()){
                    if(!resCls.getParentFile().exists()){
                        resCls.getParentFile().mkdirs();
                    }
                    try {
                        resCls.createNewFile();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        Set datasetDone = new HashSet();
        if(resCls.exists()){
            try {
                BufferedReader br = new BufferedReader(new FileReader(resCls));
                String s;
                while((s=br.readLine())!=null){
                    datasetDone.add(s.split("[,]")[0]);
                }
                br.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(resCls,true));
            //if children class turnOnCV(), uncommand below
//
            String tscProbDir = "D:/datasets/Univariate_arff/";
            BufferedReader br =new BufferedReader(new FileReader(new File("D:\\硕士培养手册\\小论文\\dataForJavaClsStep.txt")));

            boolean datasetSavePathOnce =false;
            String line;
            int count = 0;
            while((line=br.readLine())!=null){
                count++;
                if(count<-1){
                    continue;
                }

                String[] a = line.trim().split("[,]");
                datasetName = a[0];

                this.step_p = Float.parseFloat(a[1]);
                System.out.println(datasetName+","+this.step_p);
                if(!datasetSavePathOnce&&datasetSavePath.equals("theory")){
                    datasetName=datasetSavePath;
                    datasetSavePathOnce =true;
                }
                if(!datasetSavePath.equals("None")&&!datasetSavePath.equals(datasetName)){
                    continue;
                }
                System.out.println("datasetName:    "+datasetName);
                if(datasetDone.contains(datasetName)){
                    System.out.println("already done!");
                    continue;
                }

                if(!datasetName.equals(checkPoint.getDatasetName())){
                    checkPoint.initialParas(datasetName);
                }

                int trainTestFold = resample100? 100:1;

                int correctNewTotal = 0,testNewTotal = 0;
                long startTest,startTrain, end, newTime=0,newTime2=0;
                for(int foldId=0;foldId<trainTestFold;foldId++) {
                    Instances train,test;
                    if(trainTestFold==1){
                        train = ClassifierTools.loadData(tscProbDir + datasetName + "/" + datasetName + "_TRAIN");
                        test = ClassifierTools.loadData(tscProbDir + datasetName + "/" + datasetName + "_TEST");
                    }else{
                        Instances[] data = ExperimentsClean.sampleDataset(tscProbDir, datasetName, foldId);
                        train = data[0];
                        test = data[1];
                        testNewTotal += test.numInstances();
                    }
                    if(this.classifierIdentifier.split("[_]")[0].equals("DDTW")){
                        train = derf.process(train);
                        test = derf.process(test);
                    }
                    this.buildClassifier(train);

                    // validate
                    startTrain = System.nanoTime();
                    if (cv) {
                        this.loocv(train);
                        System.out.println("parameters after cv:" + this.getParamInformationString());
                    }
                    if(stepAnalyze){
                        this.datasetName = datasetName;
                        this.loocvStep(train,test); // write to file each accuracy to each step
                    }
                    newTime2 += System.nanoTime()-startTrain;

                    int correctNew = 0;
                    double pred;
                    // classification with new MSM and own 1NN
                    startTest = System.nanoTime();
                    correctNew = 0;

                    if(allowLoocv)
                        setParamsFromParamId(train,checkPoint.getValidateBest().getKey());

                    for (int i = 0; i < test.numInstances(); i++) {
                        if(i%10==0)
                            System.out.println("[Loop] Test case: " + i);
                        if(i<checkPoint.getTestNum()){
                            correctNew =checkPoint.getTestCorrect();
                            i = checkPoint.getTestNum();
                            System.out.println("[checkpoint]testing for case: "+i + "correct: " +correctNew);
                            continue;
                        }
                        if(stepAnalyze) continue;
                        if(datasetSavePath.equals("None")){ // not save path for Singularities, etc
                            pred = this.classifyInstance(test.instance(i));
                        }else{
                            Pair predAndPath = this.classifyInstanceSaveDistAndPath(test.instance(i));

                            pred = (double)predAndPath.getKey();

                            List distPathList =(List)predAndPath.getValue();
                            resCls = new File(distPathDir+"/test"+i+".txt");
                            bw.flush();
                            bw.close();
                            bw = new BufferedWriter(new FileWriter(resCls));
                            for(int k=0;k<distPathList.size();k++){
                                Pair distPath = (Pair) distPathList.get(k);
                                bw.write(distPath.getKey()+"|"+distPath.getValue()+"\n");
                            }
                            System.out.println(resCls);
                            if(datasetName.equals("theory")){
                            Pair predAndPathDist = this.classifyInstanceSaveDistAndPathDist(test.instance(i));
                            distPathList =(List)predAndPathDist.getValue();
                            resCls = new File(distPathDir+"/dist"+i+".txt");
                            bw.flush();
                            bw.close();
                            bw = new BufferedWriter(new FileWriter(resCls));
                            for(int k=0;k<distPathList.size();k++){
                                Pair distPath = (Pair) distPathList.get(k);
                                bw.write(distPath.getKey()+"|"+distPath.getValue()+"\n");
                            }
                            System.out.println(resCls);}

                        }

                        if (pred == test.instance(i).classValue()) {
                            correctNew++;
                            if(i%10==0){
                                checkPoint.setTestNum(i);
                                checkPoint.setTestCorrect(correctNew);
                                checkPoint.saveParas();
                            }
                            correctNewTotal++;
                        }
                    }
                    end = System.nanoTime();
                    newTime += end-startTest;

                    if(trainTestFold==1){
                        System.out.println("datasetName:    "+datasetName);
                        System.out.println("New acc:    "+((double)correctNew/test.numInstances()));
                        System.out.println("New timing: "+newTime);
                        if(datasetSavePath.equals("None")) {
                            String trainTime = "";
                            if(saveTrainTime){
                                trainTime += newTime2+",";
                            }
                            if (cv) {
                                bw.write(datasetName + "," + ((double) correctNew / test.numInstances()) + "," + newTime + "," +trainTime+ this.getParamInformationString() + "\n");
                            } else {
                                bw.write(datasetName + "," + ((double) correctNew / test.numInstances()) + "," + newTime + "\n");
                            }
                        }
                        bw.flush();
                    }else{
                        System.out.println("datasetName:    "+datasetName+"  FoldId:\t"+foldId);
                        System.out.println("New acc:    "+((double)correctNew/test.numInstances()));
                        System.out.println("New timing: "+(end-startTest));
                    }
                }
                if(trainTestFold != 1){
                    bw.write(datasetName+","+((double)correctNewTotal/testNewTotal)+","+newTime+"\n");
                    bw.flush();
                }
                if(!datasetSavePath.equals("None")){
                    resCls = new File("D:/硕士培养手册/小论文/results/distPath/"+totalClassName[totalClassName.length-1]+"/datasetDone.txt");
                    if(stepAnalyze)
                        resCls = new File("D:/硕士培养手册/小论文/results/step/"+totalClassName[totalClassName.length-1]+"/datasetDone.txt");
                    bw.flush();
                    bw.close();
                    bw = new BufferedWriter(new FileWriter(resCls,true));
                    if(!datasetSavePath.equals("theory"))
                        bw.write(datasetSavePath+"\n");
                    bw.flush();
                }
            }
            bw.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }


    private String getOutclassType(){
        String[] fullClassName = getClass().toString().split(" ")[1].split("[.]");
        return fullClassName[fullClassName.length-1];
    };

    public class CheckPoint{

        public void setValidateId(int validateId) {
            this.validateId = validateId;
        }

        public void setValidateBest(Pair<Integer, Double> validateBest) {
            this.validateBest = validateBest;
        }

        public void setValidateNum(int validateNum) {
            this.validateNum = validateNum;
        }

        public void setValidateCorrect(int validateCorrect) {
            this.validateCorrect = validateCorrect;
        }

        public void setTestNum(int testNum) {
            this.testNum = testNum;
        }

        public void setTestCorrect(int testCorrect) {
            this.testCorrect = testCorrect;
        }

        public void setAllPara(Properties allPara) {
            this.allPara = allPara;
        }

        public int getValidateId() {
            return validateId;
        }

        public int getValidateNum() {
            return validateNum;
        }

        public int getValidateCorrect() {
            return validateCorrect;
        }

        public Pair<Integer, Double> getValidateBest() {
            return validateBest;
        }

        public int getTestNum() {
            return testNum;
        }

        public int getTestCorrect() {
            return testCorrect;
        }

        public Properties getAllPara() {
            return allPara;
        }

        public String getDatasetName() {
            return datasetName;
        }

        public void setDatasetName(String datasetName) {
            this.datasetName = datasetName;
        }

        @Override
        public String toString() {
            return "CheckPoint{" +
                    "datasetName='" + datasetName + '\'' +
                    ", validateId=" + validateId +
                    ", validateNum=" + validateNum +
                    ", validateCorrect=" + validateCorrect +
                    ", validateBest=" + validateBest +
                    ", testNum=" + testNum +
                    ", testCorrect=" + testCorrect +
                    ", allPara=" + allPara +
                    '}';
        }

        public CheckPoint(){
            allPara = new Properties();
            loadParas();
            System.out.println(allPara);
            saveParas();
            System.out.println(this.toString());
        }

        private String datasetName = "None";
        private int validateId =-1;
        private int validateNum =-1;
        private int validateCorrect =-1;
        private Pair<Integer,Double> validateBest =new Pair<>(-1,0.0);
        private int testNum =-1;
        private int testCorrect=-1;
        Properties allPara = new Properties();


        private void loadParas(){
            if(!saveCheckPoint)  //不适用缓存，则所有参数是初始化参数
                return;
            allParasFromFile();
            System.out.println(allPara);
            setValidateId(Integer.parseInt(allPara.getProperty("validateId","-1")));
            setValidateNum(Integer.parseInt(allPara.getProperty("validateNum","-1")));
            setValidateCorrect(Integer.parseInt(allPara.getProperty("validateCorrect","-1")));
            setValidateBest(
                    new Pair(Integer.parseInt(allPara.getProperty("validateBest.Id","-1")),
                            Double.parseDouble(allPara.getProperty("validateBest.Acc","0.0"))));

            setTestNum(Integer.parseInt(allPara.getProperty("testNum","-1")));
            setTestCorrect(Integer.parseInt(allPara.getProperty("testCorrect","-1")));
            setTestCorrect(Integer.parseInt(allPara.getProperty("testCorrect","-1")));
            setDatasetName(allPara.getProperty("datasetName","None"));
        }

        private void saveParas(){
            allPara.setProperty("datasetName", datasetName);
            allPara.setProperty("validateId",String.valueOf(validateId) );
            allPara.setProperty("validateNum",String.valueOf(validateNum) );
            allPara.setProperty("validateCorrect",String.valueOf(validateCorrect) );
            allPara.setProperty("validateBest.Id",String.valueOf(validateBest.getKey()) );
            allPara.setProperty("validateBest.Acc",String.valueOf(validateBest.getValue()) );

            allPara.setProperty("testNum",String.valueOf(testNum) );
            allPara.setProperty("testCorrect",String.valueOf(testCorrect) );
            allParasToFile();
        }

        public void allParasFromFile(){
            String fileName = "UCR_2018_"+getOutclassType()+"_save.properties";
            File file = new File(fileName);
            if(file.exists()){
                try {
                    FileInputStream fi=  new FileInputStream(fileName);
                    allPara.load(fi);
                    fi.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        public void allParasToFile(){
            String fileName = "UCR_2018_"+getOutclassType()+"_save.properties";
            File file = new File(fileName);
            try {
                FileOutputStream fo=  new FileOutputStream(fileName);
                allPara.store(fo, "parameters");
                fo.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public void initialParas(String name) {
            datasetName = name;
            validateId =-1;
            validateNum =-1;
            validateCorrect =-1;
            validateBest =new Pair<>(-1,0.0);
            testNum =-1;
            testCorrect=-1;
            allPara = new Properties();
            saveParas();
        }
    }

}
