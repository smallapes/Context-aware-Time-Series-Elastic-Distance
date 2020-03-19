// written April '16 - looks good


package timeseriesweka.classifiers.ensembles.elastic_ensemble;

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
public class CWDTW1NN extends Efficient1NN{

    private double g = 0;
    private double step;
    private double[] weightVector;
    private static final double WEIGHT_MAX = 1;
    private boolean refreshWeights = true;

    public CWDTW1NN(double g,double step){
        this.g = g;
        this.step = step;
        this.classifierIdentifier = "CWDTW_1NN";
        this.allowLoocv = false;
    }

    public CWDTW1NN(){
        this.g = 0;
        this.step =0.05;
        this.classifierIdentifier = "CWDTW_1NN";
    }

    @Override
    public void buildClassifier(Instances train) throws Exception{
        this.train = train;
        this.trainGroup = null;
        this.refreshWeights = true;
    }
    
    private void initWeights(int seriesLength){
        this.weightVector = new double[seriesLength];
        double halfLength = (double)seriesLength/2;

        for(int i = 0; i < seriesLength; i++){
            weightVector[i] = WEIGHT_MAX/(1+Math.exp(-g*(i-halfLength)));
        }
        refreshWeights = false;
    }

    @SuppressWarnings("all")
    public final double distance(Instance first, Instance second, double cutoff){
        
        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
        if(first.classIndex() != first.numAttributes()-1 || second.classIndex()!=second.numAttributes()-1){
            return new WeightedDTW(g).distance(first, second,cutoff);
        }        
        
        int m = first.numAttributes()-1;
        int n = second.numAttributes()-1;
        int step = (int)((m+n)/2*this.step);
        
        if(this.refreshWeights){
            this.initWeights(m);
        }


        //create empty array
        double[][] distances = new double[m][n];
        
        //first value
        distances[0][0] = this.weightVector[0]*(first.value(0)-second.value(0))*(first.value(0)-second.value(0));
        
        //early abandon if first values is larger than cut off
        if(distances[0][0] > cutoff){
            return Double.MAX_VALUE;
        }
        
        //top row
        for(int j=1;j<n;j++){
            int i = 0;
            distances[0][j] = distances[0][j-1]+this.weightVector[j]*(Math.pow(first.value(0)-second.value(j),2)+
                    Math.pow(first.value(Math.max(i-step,0))-second.value(Math.max(j-step,0)),2)+
                    Math.pow(first.value(Math.min(i+step,m-1))-second.value(Math.min(j+step,n-1)),2))/3; //edited by Jay
        }

        //first column
        for(int i=1;i<m;i++){
            int j =0;
            distances[i][0] = distances[i-1][0]+this.weightVector[i]*(Math.pow(first.value(i)-second.value(0),2)+
                    Math.pow(first.value(Math.max(i-step,0))-second.value(Math.max(j-step,0)),2)+
                    Math.pow(first.value(Math.min(i+step,m-1))-second.value(Math.min(j+step,n-1)),2))/3; //edited by Jay
        }
        
        //warp rest
        double minDistance;
        for(int i = 1; i<m; i++){
            boolean overflow = true;
            
            for(int j = 1; j<n; j++){
                //calculate distances
                minDistance = Math.min(distances[i][j-1], Math.min(distances[i-1][j], distances[i-1][j-1]));
                distances[i][j] = minDistance+this.weightVector[Math.abs(i-j)] *
                            (Math.pow(first.value(i)-second.value(j),2)+
                             Math.pow(first.value(Math.max(i-step,0))-second.value(Math.max(j-step,0)),2)+
                             Math.pow(first.value(Math.min(i+step,m-1))-second.value(Math.min(j+step,n-1)),2))/3;
                
                if(overflow && distances[i][j] < cutoff){
                    overflow = false; // because there's evidence that the path can continue
                }
            }
            
            //early abandon
            if(overflow){
                return Double.MAX_VALUE;
            }
        }
        return distances[m-1][n-1];
        
        
    }
    

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void runComparison() throws Exception{
        String tscProbDir = "D:/datasets/Univariate_arff/";
        
//        String datasetName = "ItalyPowerDemand";
//        String datasetName = "GunPoint";
//        String datasetName = "Wine";
        String datasetName = "Beef";
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
        CWDTW1NN dtwNew = new CWDTW1NN(r,0.05);
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

    /**
     * Author :tcl
     * validate g
     * step =0.05
     */
    public static void UCR2018(){
        boolean cv = true;
        Efficient1NN dtwNew = new CWDTW1NN();
        //dtwNew.turnOnCV();
        //this.cparamId = 10;
        dtwNew.cparamId = 5;
        dtwNew.UCR2018Cls(cv);
    }

    public static void main(String[] args) throws Exception{
        for(int i = 0; i < 1; i++){
//            runComparison();
            UCR2018();
        }

//        Instances train = ClassifierTools.loadData("C:/users/sjx07ngu/dropbox/tsc problems/SonyAiboRobotSurface1/SonyAiboRobotSurface1_TRAIN");
//
//        Instance one, two;
//        one = train.firstInstance();
//        two = train.lastInstance();
//        WeightedDTW wdtw;
//        CWDTW1NN wnn = new CWDTW1NN();
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
        this.g = (double)paramId/this.cparamId;
        refreshWeights = true;
    }

    @Override
    public String getParamInformationString() {
        return this.g+","+this.step;
    }
    
    public String toString(){
        return "this weight: "+this.g;
    }


    
    
    
}
