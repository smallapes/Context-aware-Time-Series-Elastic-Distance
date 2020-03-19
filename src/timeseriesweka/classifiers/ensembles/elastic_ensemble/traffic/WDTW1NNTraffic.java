// written April '16 - looks good


package timeseriesweka.classifiers.ensembles.elastic_ensemble.traffic;

import javafx.util.Pair;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.Efficient1NN;
import timeseriesweka.elastic_distance_measures.DTW;
import timeseriesweka.elastic_distance_measures.WeightedDTW;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
//import efficient_standalone_classifiers.Eff

/**
 *
 * @author sjx07ngu
 */
public class WDTW1NNTraffic extends Efficient1NNTraffic{

    private double g = 0;

    private double[] weightVector;
    private static final double WEIGHT_MAX = 1;
    private boolean refreshWeights = true;
    public static double step = 0.0;
    private double r = 0.1;

    public WDTW1NNTraffic(double g){
        this.g = g;
        this.classifierIdentifier = "WDTW_1NN";
        this.allowLoocv = false;
    }

    public WDTW1NNTraffic(){
        this.g = 0;
        this.classifierIdentifier = "WDTW_1NN";
    }
    
    private void initWeights(int seriesLength){
        this.weightVector = new double[seriesLength];
        double halfLength = (double)seriesLength/2;

        for(int i = 0; i < seriesLength; i++){
            weightVector[i] = WEIGHT_MAX/(1+Math.exp(-g*(i-halfLength))); //=1
        }
        refreshWeights = false;
    }

    //next: 计算路径的长度
    public double Lp(double x){
        return Math.abs(x);
    }
    //next: 计算路径的长度
    public double Lp(double x,int p ){
        return Math.abs(x);
    }

    public final double distance(Instance first, Instance second, double cutoff){

        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
        if(first.classIndex() != first.numAttributes()-1 || second.classIndex()!=second.numAttributes()-1){
            return new WeightedDTW(g).distance(first, second,cutoff);
        }

        int m = first.numAttributes()-1;
        int n = second.numAttributes()-1;
        if(isTraffic){
            m = (int)Math.floor(first.value(trafficFeatLen+1));
            n = (int)Math.floor(second.value(trafficFeatLen+1));
        }

        if(this.refreshWeights){
            this.initWeights(600);
        }
//        if(isTraffic){
//            this.initWeights(Math.max(m,n));
//        }


        //create empty array
        double[][] distances = new double[m][n];

        //first value
        distances[0][0] = this.weightVector[0]*Lp(first.value(0)-second.value(0));

        //early abandon if first values is larger than cut off
//        if(distances[0][0] > cutoff){
//            return Double.MAX_VALUE;
//        }


        //first column
        for(int i=1;i<m;i++){
            distances[i][0] = distances[i-1][0]+this.weightVector[i]*Lp(first.value(i)-second.value(0)); //edited by Jay
        }
        //top row
        for(int j=1;j<n;j++){
            distances[0][j] = distances[0][j-1]+this.weightVector[j]*Lp(first.value(0)-second.value(j)); //edited by Jay
        }

        //warp rest
        double minDistance;
        for(int i = 1; i<m; i++){
            boolean overflow = true;

            for(int j = 1; j<n; j++){
                //calculate distances
                minDistance = Math.min(distances[i][j-1], Math.min(distances[i-1][j], distances[i-1][j-1]));
                distances[i][j] = minDistance+this.weightVector[Math.abs(i-j)] *Lp(first.value(i)-second.value(j));

                if(overflow && distances[i][j] < cutoff){
                    overflow = false; // because there's evidence that the path can continue
                }
            }

            //early abandon
//            if(overflow){
//                return Double.MAX_VALUE;
//            }
        }
        if(distances[m-1][n-1]<1e-10){
            System.out.println();
        }
        return distances[m-1][n-1];


    }

//    public final double distance(Instance first, Instance second, double cutoff){
//
//        // base case - we're assuming class val is last. If this is true, this method is fine,
//        // if not, we'll default to the DTW class
//        if(first.classIndex() != first.numAttributes()-1 || second.classIndex()!=second.numAttributes()-1){
//            DTW temp = new DTW();
//            temp.setR(r);
//            return temp.distance(first, second,cutoff);
//        }
//
//        double minDist;
//        double minDistLen; //next : minDist出现位置
//        boolean tooBig;
//
//        int n = first.numAttributes()-1;
//        int m = second.numAttributes()-1;
//        if(isTraffic){
//            n = (int)Math.floor(first.value(trafficFeatLen+1));
//            m = (int)Math.floor(second.value(trafficFeatLen+1));
//        }
//
//        int step = (int)((m+n)/2*this.step);
//        //int step = (m+n)/40;
//        /*  Parameter 0<=r<=1. 0 == no warp, 1 == full warp
//         generalised for variable window size
//         * */
////        int windowSize = getWindowSize(n);
//        int windowSize = Math.max(m,n);
////Extra memory than required, could limit to windowsize,
////        but avoids having to recreate during CV
////for varying window sizes
//        double[][] matrixD = new double[n][m];
//
//        // next: 修改各种路径位置
//        double[][] matrixDLen = new double[n][m];
//
//        /*
//         //Set boundary elements to max.
//         */
//        int start, end;
//        for (int i = 0; i < n; i++) {
//            start = windowSize < i ? i - windowSize : 0;
//            end = i + windowSize + 1 < m ? i + windowSize + 1 : m;
//            for (int j = start; j < end; j++) {
//                matrixD[i][j] = Double.MAX_VALUE;
//            }
//        }
//        for(int i=0,j=0;i<1;i++){
//            matrixD[0][0] = (Lp((first.value(i) - second.value(j)),2)+
//                    Lp((first.value(Math.max(0,i-step)) - second.value(Math.max(0,j-step))),2)+
//                    Lp((first.value(Math.min(n-1,i+step)) - second.value(Math.min(m-1,j+step))),2)/3);
//            matrixDLen[0][0] = 1; //next
//        }
//        //a is the longer series.
////Base cases for warping 0 to all with max interval	r
////Warp first.value(0] onto all second.value(1]...second.value(r+1]
//        for (int j = 1; j < windowSize && j < m; j++) {
//            int i = 0;
//            matrixD[0][j] = matrixD[0][j - 1] + (Lp((first.value(i) - second.value(j)),2)+
//                    Lp((first.value(Math.max(0,i-step)) - second.value(Math.max(0,j-step))),2)+
//                    Lp((first.value(Math.min(n-1,i+step)) - second.value(Math.min(m-1,j+step))),2))/3;
//            matrixDLen[0][j] = matrixD[0][j - 1]+1; //next
//        }
//
////	Warp second.value(0] onto all first.value(1]...first.value(r+1]
//        for (int i = 1; i < windowSize && i < n; i++) {
//            int j = 0;
//            matrixD[i][0] = matrixD[i - 1][0] + (Lp((first.value(i) - second.value(j)),2)+
//                    Lp((first.value(Math.max(0,i-step)) - second.value(Math.max(0,j-step))),2)+
//                    Lp((first.value(Math.min(n-1,i+step)) - second.value(Math.min(m-1,j+step))),2))/3;
//            matrixDLen[i][0] = matrixDLen[i - 1][0]+1;//next
//        }
////Warp the rest,
//        for (int i = 1; i < n; i++) {
//            tooBig = true;
//            start = windowSize < i ? i - windowSize + 1 : 1;
//            end = i + windowSize < m ? i + windowSize : m;
//            for (int j = start; j < end; j++) {
//                minDist = matrixD[i][j - 1];
//                minDistLen = matrixDLen[i][j - 1];
//                if (matrixD[i - 1][j] < minDist) {
//                    minDist = matrixD[i - 1][j];
//                    minDistLen = matrixDLen[i - 1][j];
//                }
//                if (matrixD[i - 1][j - 1] < minDist) {
//                    minDist = matrixD[i - 1][j - 1];
//                    minDistLen = matrixDLen[i - 1][j - 1];
//                }
//                matrixD[i][j] = minDist + (Lp((first.value(i) - second.value(j)),2)+
//                        Lp((first.value(Math.max(0,i-step)) - second.value(Math.max(0,j-step))),2)+
//                        Lp((first.value(Math.min(n-1,i+step)) - second.value(Math.min(m-1,j+step))),2))/3;
//
//                matrixDLen[i][j] = minDistLen+1;
//                if (tooBig && matrixD[i][j] < cutoff) {
//                    tooBig = false;
//                }
//            }
//            //Early abandon
////            if (tooBig) {
////                return Double.MAX_VALUE;
////            }
//        }
////Find the minimum distance at the end points, within the warping window.
//
//        return matrixD[n-1][m-1]/matrixDLen[n-1][m-1];
//    }


    public final Pair<Double,double[][]> distanceSavePath(Instance first, Instance second, double cutoff){

        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
//        if(first.classIndex() != first.numAttributes()-1 || second.classIndex()!=second.numAttributes()-1){
//            return new WeightedDTW(g).distance(first, second,cutoff);
//        }

        int m = first.numAttributes()-1;
        int n = second.numAttributes()-1;

        if(this.refreshWeights){
            this.initWeights(m);
        }


        //create empty array
        double[][] distances = new double[m][n];

        //first value
        distances[0][0] = this.weightVector[0]*(first.value(0)-second.value(0))*(first.value(0)-second.value(0));

//        //early abandon if first values is larger than cut off
//        if(distances[0][0] > cutoff){
//            return Double.MAX_VALUE;
//        }

        //top row
        for(int i=1;i<n;i++){
            distances[0][i] = distances[0][i-1]+this.weightVector[i]*(first.value(0)-second.value(i))*(first.value(0)-second.value(i)); //edited by Jay
        }

        //first column
        for(int i=1;i<m;i++){
            distances[i][0] = distances[i-1][0]+this.weightVector[i]*(first.value(i)-second.value(0))*(first.value(i)-second.value(0)); //edited by Jay
        }

        //warp rest
        double minDistance;
        for(int i = 1; i<m; i++){
            boolean overflow = true;

            for(int j = 1; j<n; j++){
                //calculate distances
                minDistance = Math.min(distances[i][j-1], Math.min(distances[i-1][j], distances[i-1][j-1]));
                distances[i][j] = minDistance+this.weightVector[Math.abs(i-j)] *(first.value(i)-second.value(j))*(first.value(i)-second.value(j));

                if(overflow && distances[i][j] < cutoff){
                    overflow = false; // because there's evidence that the path can continue
                }
            }

//            //early abandon
//            if(overflow){
//                return Double.MAX_VALUE;
//            }
        }
        return new Pair<>(distances[m-1][n-1],distances);


    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void runComparison() throws Exception{
        String tscProbDir = "D:/datasets/Univariate_arff/";
        
//        String datasetName = "ItalyPowerDemand";
        String datasetName = "GunPoint";
//        String datasetName = "Beef";
//        String datasetName = "Coffee";
//        String datasetName = "SonyAiboRobotSurface1";

        double r = 0.1;
        Instances train = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TEST");
        
        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        WeightedDTW oldDtw = new WeightedDTW(r);
        knn.setDistanceFunction(oldDtw);
        knn.buildClassifier(train);
        
        // new version
        WDTW1NNTraffic dtwNew = new WDTW1NNTraffic(r);
        dtwNew.buildClassifier(train);
        
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

    @Override
    public void buildClassifier(Instances train) throws Exception{
        this.train = train;
        this.trainGroup = null;
        this.refreshWeights = true;
    }

    /**
     * Author :tcl
     * validate g
     */
    public static void UCR2018(){
        boolean cv = true;
        Efficient1NNTraffic dtwNew = new WDTW1NNTraffic();
        //dtwNew.turnOnCV();
        //this.cparamId = 10;
        dtwNew.UCR2018Cls(cv);
    }
    /**
     *
     */
    public static void UCR2018SavePath(){
        boolean cv = true;
        boolean resample100 =false;
        Efficient1NNTraffic dtwNew = new WDTW1NNTraffic();
        dtwNew.UCR2018Cls(cv, resample100,"Wine");
    }

    /**
     *
     */
    public static void UCR2018SavePathTheory(){
        boolean cv =false;
        boolean resample100 =false;
        Efficient1NNTraffic dtwNew = new WDTW1NNTraffic(0.25);
        dtwNew.UCR2018Cls(cv, resample100,"theory");
    }

    /**
     *
     */
    public static void UCR2018SavePathAll(){
        boolean cv = true;
        boolean resample100 =false;
        Efficient1NNTraffic dtwNew = new WDTW1NNTraffic();
        dtwNew.UCR2018ClsSavePathAll(cv, resample100);
        UCR2018SavePath();
    }
    public static void main(String[] args) throws Exception{
        for(int i = 0; i < 1; i++){
//            runComparison();
            UCR2018();
//            UCR2018SavePathTheory();
//            UCR2018SavePathAll();
        }

//        Instances train = ClassifierTools.loadData("C:/users/sjx07ngu/dropbox/tsc problems/SonyAiboRobotSurface1/SonyAiboRobotSurface1_TRAIN");
//
//        Instance one, two;
//        one = train.firstInstance();
//        two = train.lastInstance();
//        WeightedDTW wdtw;
//        WDTW1NN wnn = new WDTW1NN();
//        double g;
//        for(int paramId = 0; paramId < 100; paramId++){
//            g = (double)paramId/100;
//            wdtw = new WeightedDTW(g);
//
//            wnn.setParamsFromParamId(train, paramId);
//            System.out.print(wdtw.distance(one, two)+"\t");
//            System.out.println(wnn.distance(one, two,Double.MAX_VALUE));
//
//        }


    }

    @Override
    public void setParamsFromParamId(Instances train, int paramId) {
        this.g = (double)paramId/100;
        refreshWeights = true;
    }

    @Override
    public String getParamInformationString() {
        return this.g+",";
    }
    
    public String toString(){
        return "this weight: "+this.g;
    }


    
    
    
}
