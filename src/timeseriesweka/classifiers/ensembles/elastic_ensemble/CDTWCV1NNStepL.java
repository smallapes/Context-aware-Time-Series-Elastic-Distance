package timeseriesweka.classifiers.ensembles.elastic_ensemble;

import timeseriesweka.elastic_distance_measures.DTW;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

/**
 *
 * @author tcl
 * Date: 19/12/29
 * to observe what effect different step and L have on the accuracy ?
 * have no predict accuracy record
 *
 * step = [0, 0.01,...,0.99]
 *  L = [1, 2,..., 10]
 */
public class CDTWCV1NNStepL extends Efficient1NN {
    private double r;
    private double step;
    private int L;
    private double[] steps ={0,0.001,0.0025,0.005,0.01,0.02,0.04,0.05,0.08,0.16};


    /**
     * Constructor with specified window size (between 0 and 1). When a window
     * size is specified, cross-validation methods will become inactive for this
     * object.
     *
     * Note: if window = 1, classifierIdentifier will be DTW_R1_1NN; other
     * window sizes will results in cId of DTW_Rn_1NN instead. This information
     * is used for any file writing
     * @param r
     */
    public CDTWCV1NNStepL(double r){
        this.allowLoocv = false;
        this.r = r;
        if(r!=1){
            this.classifierIdentifier = "DTW_Rn_1NN";
        }else{
            this.classifierIdentifier = "DTW_R1_1NN";
        }
    }

    /**
     * A default constructor. Sets the window to 1 (100%), but allows for the
     * option of cross-validation if the relevant method is called.
     *
     * classifierIdentifier is initially set to DTW_R1_1NN, but will
     * update automatically to DTW_Rn_1NN if loocv is called
     */
    public CDTWCV1NNStepL(){
        this.r = 1;
        this.step =0.05;
        this.classifierIdentifier = "CDTWCV_R1_1NN";
    }
    public void setWindow(double w){ r=w;}

    public void turnOffCV(){
        this.allowLoocv = false;
    }
     public void turnOnCV(){
        this.allowLoocv = true;
    }

    @Override
    public double[] loocv(Instances train) throws Exception{
        if(this.allowLoocv==true && this.classifierIdentifier.contains("R1")){
            this.classifierIdentifier=this.classifierIdentifier.replace("R1", "Rn");
        }
        return super.loocv(train);
    }

    @Override
    public double[] loocv(Instances[] train) throws Exception{
        if(this.allowLoocv==true && this.classifierIdentifier.contains("R1")){
            this.classifierIdentifier=this.classifierIdentifier.replace("R1", "Rn");
        }
        return super.loocv(train);
    }

    final public int getWindowSize(int n){
        int w=(int)(r*n);   //Rounded down.
                //No Warp, windowSize=1
        if(w<1) w=1;
                //Full Warp : windowSize=n, otherwise scale between
        else if(w<n)
                w++;
        return w;
    }



//    public double classifyInstance(Instance instance) throws Exception{
//        if(isDerivative){
//            Instances temp = new Instances(instance.dataset(),1);
//            temp.add(instance);
//            temp = new DerivativeFilter().process(temp);
//            return classifyInstance(temp.instance(0));
//        }
//        return super.classifyInstance(instance);
//    }
    @SuppressWarnings("all")
    public final double distance(Instance first, Instance second, double cutoff){

        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
        if(first.classIndex() != first.numAttributes()-1 || second.classIndex()!=second.numAttributes()-1){
            DTW temp = new DTW();
            temp.setR(r);
            return temp.distance(first, second,cutoff);
        }

        double minDist;
        boolean tooBig;

        int n = first.numAttributes()-1;
        int m = second.numAttributes()-1;
        int step = (int)((m+n)/2*this.step);
        //int step = (m+n)/40;
            /*  Parameter 0<=r<=1. 0 == no warp, 1 == full warp
             generalised for variable window size
             * */
        int windowSize = getWindowSize(n);
    //Extra memory than required, could limit to windowsize,
    //        but avoids having to recreate during CV
    //for varying window sizes
        double[][] matrixD = new double[n][m];

            /*
             //Set boundary elements to max.
             */
        int start, end;
        for (int i = 0; i < n; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = i + windowSize + 1 < m ? i + windowSize + 1 : m;
            for (int j = start; j < end; j++) {
                matrixD[i][j] = Double.MAX_VALUE;
            }
        }
        int l =0;
        for(int i=0,j=0;i<1;i++){

            matrixD[i][j] = Math.pow((first.value(i) - second.value(j)),2);
            for(l=1;l<=this.L;l++){
                matrixD[i][j] += Math.pow((first.value(Math.max(0,i-step*l)) - second.value(Math.max(0,j-step*l))),2)+
                        Math.pow((first.value(Math.min(n-1,i+step*l)) - second.value(Math.min(m-1,j+step*l))),2);
            }
            matrixD[i][j] /= (2*this.L+1);

        }
        //a is the longer series.
    //Base cases for warping 0 to all with max interval	r
    //Warp first.value(0] onto all second.value(1]...second.value(r+1]
        for (int j = 1; j < windowSize && j < m; j++) {
            int i = 0;

            matrixD[i][j] = Math.pow((first.value(i) - second.value(j)),2);
            for(l=1;l<=this.L;l++){
                matrixD[i][j] += Math.pow((first.value(Math.max(0,i-step*l)) - second.value(Math.max(0,j-step*l))),2)+
                        Math.pow((first.value(Math.min(n-1,i+step*l)) - second.value(Math.min(m-1,j+step*l))),2);
            }
            matrixD[i][j] /= (2*this.L+1);
            matrixD[i][j] +=matrixD[i][j-1];

        }

    //	Warp second.value(0] onto all first.value(1]...first.value(r+1]
        for (int i = 1; i < windowSize && i < n; i++) {
            int j = 0;

            matrixD[i][j] = Math.pow((first.value(i) - second.value(j)),2);
            for(l=1;l<=this.L;l++){
                matrixD[i][j] += Math.pow((first.value(Math.max(0,i-step*l)) - second.value(Math.max(0,j-step*l))),2)+
                        Math.pow((first.value(Math.min(n-1,i+step*l)) - second.value(Math.min(m-1,j+step*l))),2);
            }
            matrixD[i][j] /= (2*this.L+1);
            matrixD[i][j] +=matrixD[i-1][j];

        }
    //Warp the rest,
        for (int i = 1; i < n; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = i + windowSize < m ? i + windowSize : m;
            for (int j = start; j < end; j++) {
                minDist = matrixD[i][j - 1];
                if (matrixD[i - 1][j] < minDist) {
                    minDist = matrixD[i - 1][j];
                }
                if (matrixD[i - 1][j - 1] < minDist) {
                    minDist = matrixD[i - 1][j - 1];
                }

                matrixD[i][j] = Math.pow((first.value(i) - second.value(j)),2);
                for(l=1;l<=this.L;l++){
                    matrixD[i][j] += Math.pow((first.value(Math.max(0,i-step*l)) - second.value(Math.max(0,j-step*l))),2)+
                            Math.pow((first.value(Math.min(n-1,i+step*l)) - second.value(Math.min(m-1,j+step*l))),2);
                }
                matrixD[i][j] /= (2*this.L+1);
                matrixD[i][j] += minDist;

                if (tooBig && matrixD[i][j] < cutoff) {
                    tooBig = false;
                }
            }
            //Early abandon
            if (tooBig) {
                return Double.MAX_VALUE;
            }
        }
    //Find the minimum distance at the end points, within the warping window.
        return matrixD[n-1][m-1];
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
       double[] res=new double[instance.numClasses()];
       int r=(int)classifyInstance(instance);
       res[r]=1;
       return res;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    /**
     Author:tcl
     validation for best parameter step
     public static void runComparison() throws Exception{};
     dtwNew.turnOnCV();
     dtwNew.cparamId = 10;
     dtwNew.loocv(train);
     System.out.println(dtwNew.getParamInformationString());

     combine with children class's
     public void setParamsFromParamId(Instances train, int paramId);
     */
    public static void runComparison() throws Exception{
        String tscProbDir = "D:/datasets/Univariate_arff/";
        //String tscProbDir = "C:/users/sjx07ngu/Dropbox/TSC Problems/";

//        String datasetName = "ItalyPowerDemand";
        String datasetName = "Beef";
//        String datasetName = "Beef";
//        String datasetName = "Coffee";
//        String datasetName = "SonyAiboRobotSurface1";

        double r = 0.1;
        Instances train = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TEST");
        
        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        DTW oldDtw = new DTW();
        oldDtw.setR(r);
        knn.setDistanceFunction(oldDtw);
        knn.buildClassifier(train);
        
        // new version
        CDTWCV1NNStepL dtwNew = new CDTWCV1NNStepL(r);
        dtwNew.buildClassifier(train);

        /**Author:tcl
         * validation fot best parameter step
        */
        dtwNew.turnOnCV();
        dtwNew.cparamId = 10;
        dtwNew.loocv(train);
        System.out.println(dtwNew.getParamInformationString());

        int correctOld = 0;
        int correctNew = 0;
        
        long start, end, oldTime, newTime;
        double pred;
               
        // classification with old MSM class and kNN
        start = System.nanoTime();
        
        correctOld = 0;
        for(int i = 0; i < test.numInstances(); i++){
            pred = knn.classifyInstance(test.instance(i));
            if(pred==test.instance(i).classValue()){
                correctOld++;
            }
        }
        end = System.nanoTime();
        oldTime = end-start;
        
        // classification with new MSM and own 1NN
        start = System.nanoTime();
        correctNew = 0;
        for(int i = 0; i < test.numInstances(); i++){
            pred = dtwNew.classifyInstance(test.instance(i));
            if(pred==test.instance(i).classValue()){
                correctNew++;
            }
        }
        end = System.nanoTime();
        newTime = end-start;
        
        System.out.println("Comparison of MSM: "+datasetName);
        System.out.println("==========================================");
        System.out.println("Old acc:    "+((double)correctOld/test.numInstances()));
        System.out.println("New acc:    "+((double)correctNew/test.numInstances()));
        System.out.println("Old timing: "+oldTime);
        System.out.println("New timing: "+newTime);
        System.out.println("Relative Performance: " + ((double)newTime/oldTime));
    }

    /**
     * @Author :tcl
     * validate for step [0,0.01,...,0.09]
     */
    /**
     * Author :tcl
     */
    public static void UCR2018(){
        boolean cv = true;
        Efficient1NN dtwNew = new CDTWCV1NNStepL();
        dtwNew.cparamId = 10;
        dtwNew.UCR2018Cls(cv);
    }

    /**
     * @Author :tcl
     * validate for step [0,0.01,...,0.09]
     */
    /**
     * Author :tcl
     */
    public static void UCR2018StepAnalyze(){
        Efficient1NN dtwNew = new CDTWCV1NNStepL();
        dtwNew.cparamId = 1000;

        try {
            BufferedReader br =new BufferedReader(new FileReader(new File("D:\\硕士培养手册\\小论文\\dataForJavaCls.txt")));
            String datasetName;
            while((datasetName=br.readLine())!=null){
                dtwNew.UCR2018Cls(false,false,
                        datasetName,true);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    public static void main(String[] args) throws Exception{
        for(int i = 0; i < 1; i++){
            //runComparison();
//            UCR2018();
            UCR2018StepAnalyze();
        }
    }

    @Override
    public void setParamsFromParamId(Instances train, int paramId) {
        if(this.allowLoocv){
            if(this.classifierIdentifier.contains("R1")){
                this.classifierIdentifier=this.classifierIdentifier.replace("R1", "Rn");
            }
            this.step = 1 * ((double)paramId%100/100.0);
            this.L = (int)Math.floor(paramId/100)+1;
            System.out.println(paramId+","+this.getParamInformationString());
        }else{
            throw new RuntimeException("Warning: trying to set parameters of a fixed window DTW");
        }
    }

    @Override
    public String getParamInformationString() {
        return this.r+","+this.step+","+this.L;
    }


}
