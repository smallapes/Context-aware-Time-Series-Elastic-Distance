// checked April '16

package timeseriesweka.classifiers.ensembles.elastic_ensemble.traffic;


import javafx.util.Pair;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.Efficient1NN;
import timeseriesweka.elastic_distance_measures.MSMDistance;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author sjx07ngu
 */
//public class MSM1NN implements Classifier{
public class CMSMTraffic1NN extends Efficient1NNTraffic{

    private Instances train = null;
    private double c = 0;
    private double step;

    protected static double[] msmParams = {
        // <editor-fold defaultstate="collapsed" desc="hidden for space">
        0.01,
        0.01375,
        0.0175,
        0.02125,
        0.025,
        0.02875,
        0.0325,
        0.03625,
        0.04,
        0.04375,
        0.0475,
        0.05125,
        0.055,
        0.05875,
        0.0625,
        0.06625,
        0.07,
        0.07375,
        0.0775,
        0.08125,
        0.085,
        0.08875,
        0.0925,
        0.09625,
        0.1,
        0.136,
        0.172,
        0.208,
        0.244,
        0.28,
        0.316,
        0.352,
        0.388,
        0.424,
        0.46,
        0.496,
        0.532,
        0.568,
        0.604,
        0.64,
        0.676,
        0.712,
        0.748,
        0.784,
        0.82,
        0.856,
        0.892,
        0.928,
        0.964,
        1,
        1.36,
        1.72,
        2.08,
        2.44,
        2.8,
        3.16,
        3.52,
        3.88,
        4.24,
        4.6,
        4.96,
        5.32,
        5.68,
        6.04,
        6.4,
        6.76,
        7.12,
        7.48,
        7.84,
        8.2,
        8.56,
        8.92,
        9.28,
        9.64,
        10,
        13.6,
        17.2,
        20.8,
        24.4,
        28,
        31.6,
        35.2,
        38.8,
        42.4,
        46,
        49.6,
        53.2,
        56.8,
        60.4,
        64,
        67.6,
        71.2,
        74.8,
        78.4,
        82,
        85.6,
        89.2,
        92.8,
        96.4,
        100// </editor-fold>
    };

    public CMSMTraffic1NN(){
        this.c = 0.1;
        this.step = 0.01;
        this.cparamId = 100;
        this.classifierIdentifier = "CMSM_1NN";
    }

    public CMSMTraffic1NN(double c, double step){
        this.step = step;
        this.c = c;
        this.classifierIdentifier = "CMSM_1NN";
        this.allowLoocv = false;
    }
    
   
    
    @SuppressWarnings("all")
    public double distance(Instance first, Instance second, double cutOffValue) {
        
        // need to remove class index/ignore
        // simple check - if its last, ignore it. If it's not last, copy the instances, remove that attribue, and then call again
        
        // Not particularly efficient in the latter case, but a reasonable assumption to make here since all of the UCR/UEA problems
        // match that format. 
        
        int n, m;
        if(first.classIndex()==first.numAttributes()-1 && second.classIndex()==second.numAttributes()-1){
            n = first.numAttributes()-1;
            m = second.numAttributes()-1;
        }else{            
            // default case, use the original MSM class (horrible efficiency, but just in as a fail safe for edge-cases) 
            System.err.println("Warning: class designed to use problems with class index as last attribute. Defaulting to original MSM distance");
            MSMDistance msm = new MSMDistance(this.c);
            return new MSMDistance(this.c).distance(first, second);
        }
        if(isTraffic){
            n = (int)Math.floor(first.value(trafficFeatLen+1));
            m = (int)Math.floor(second.value(trafficFeatLen+1));
        }

        int step = (int)((n+m)/2*this.step);

        double[][] cost = new double[n][m];

        // Initialization
        cost[0][0] = Math.abs(first.value(0) - second.value(0));
        for (int i = 1; i < n; i++) {
            List list_i = new ArrayList<Double>();
            list_i.add(first.value(i));
            list_i.add(first.value(Math.max(i-step,0)));
            list_i.add(first.value(Math.min(i+step,n-1)));

            List list_j = new ArrayList<Double>();
            list_j.add(second.value(0));
            list_j.add(second.value(0));
            list_j.add(second.value(Math.min(step,m-1)));

            List list_im1 = new ArrayList<Double>();
            list_im1.add(first.value(i-1));
            list_im1.add(first.value(Math.max(i-1-step,0)));
            list_im1.add(first.value(Math.min(i-1+step,n-1)));

            cost[i][0] = cost[i - 1][0] + calcualteCost(list_i,list_im1, list_j);
        }
        for (int j = 1; j < m; j++) {
            List list_i = new ArrayList<Double>();
            list_i.add(first.value(0));
            list_i.add(first.value(0));
            list_i.add(first.value(Math.min(step,n-1)));

            List list_j = new ArrayList<Double>();
            list_j.add(second.value(j));
            list_j.add(second.value(Math.max(j-step,0)));
            list_j.add(second.value(Math.min(j+step,m-1)));

            List list_jm1 = new ArrayList<Double>();
            list_jm1.add(second.value(j-1));
            list_jm1.add(second.value(Math.max(j-1-step,0)));
            list_jm1.add(second.value(Math.min(j-1+step,m-1)));
            cost[0][j] = cost[0][j - 1] + calcualteCost(list_j, list_i, list_jm1);
        }

        // Main Loop
        double min;
        for (int i = 1; i < n; i++) {
            min = cutOffValue;
            for (int j = 1; j < m; j++) {
                double d1, d2, d3;

                d1 = cost[i - 1][j - 1] + Math.sqrt((Math.pow(first.value(i) - second.value(j),2)+
                                                    Math.pow(first.value(Math.max(i-step,0))-second.value(Math.max(j-step,0)),2)+
                                                    Math.pow(first.value(Math.min(i+step,n-1))-second.value(Math.min(j+step,m-1)),2))/3
                                                    );
                List list_i = new ArrayList<Double>();
                list_i.add(first.value(i));
                list_i.add(first.value(Math.max(i-step,0)));
                list_i.add(first.value(Math.min(i+step,n-1)));

                List list_j = new ArrayList<Double>();
                list_j.add(second.value(j));
                list_j.add(second.value(Math.max(j-step,0)));
                list_j.add(second.value(Math.min(j+step,m-1)));

                List list_im1 = new ArrayList<Double>();
                list_im1.add(first.value(i-1));
                list_im1.add(first.value(Math.max(i-step-1,0)));
                list_im1.add(first.value(Math.min(i+step-1,n-1)));

                List list_jm1 = new ArrayList<Double>();
                list_jm1.add(second.value(j-1));
                list_jm1.add(second.value(Math.max(j-1-step,0)));
                list_jm1.add(second.value(Math.min(j-1+step,m-1)));

                d2 = cost[i - 1][j] + calcualteCost(list_i, list_im1, list_j);
                d3 = cost[i][j - 1] + calcualteCost(list_j, list_i, list_jm1);
                cost[i][j] = Math.min(d1, Math.min(d2, d3));
                
//                if(cost[i][j] >=cutOffValue){
//                    cost[i][j] = Double.MAX_VALUE;
//                }
                
                if(cost[i][j] < min){
                    min = cost[i][j];
                }
            }
//            if(min >= cutOffValue){
//                return Double.MAX_VALUE;
//            }
        }
        // Output
        return cost[n - 1][m - 1];
    }

    @SuppressWarnings("all")
    public final Pair<Double,double[][]> distanceSavePath(Instance first, Instance second, double cutOffValue) {

        // need to remove class index/ignore
        // simple check - if its last, ignore it. If it's not last, copy the instances, remove that attribue, and then call again

        // Not particularly efficient in the latter case, but a reasonable assumption to make here since all of the UCR/UEA problems
        // match that format.

        int n, m;
//        if(first.classIndex()==first.numAttributes()-1 && second.classIndex()==second.numAttributes()-1){
            n = first.numAttributes()-1;
            m = second.numAttributes()-1;

        if(isTraffic){
            n = (int)Math.floor(first.value(trafficFeatLen+1));
            m = (int)Math.floor(second.value(trafficFeatLen+1));
        }
//        }else{
//            // default case, use the original MSM class (horrible efficiency, but just in as a fail safe for edge-cases)
//            System.err.println("Warning: class designed to use problems with class index as last attribute. Defaulting to original MSM distance");
//            MSMDistance msm = new MSMDistance(this.c);
//            return new MSMDistance(this.c).distance(first, second);
//        }
        int step = (int)((n+m)/2*this.step);

        double[][] cost = new double[n][m];

        // Initialization
        cost[0][0] = Math.abs(first.value(0) - second.value(0));
        for (int i = 1; i < n; i++) {
            List list_i = new ArrayList<Double>();
            list_i.add(first.value(i));
            list_i.add(first.value(Math.max(i-step,0)));
            list_i.add(first.value(Math.min(i+step,n-1)));

            List list_j = new ArrayList<Double>();
            list_j.add(second.value(0));
            list_j.add(second.value(0));
            list_j.add(second.value(Math.min(step,m-1)));

            List list_im1 = new ArrayList<Double>();
            list_im1.add(first.value(i-1));
            list_im1.add(first.value(Math.max(i-1-step,0)));
            list_im1.add(first.value(Math.min(i-1+step,n-1)));

            cost[i][0] = cost[i - 1][0] + calcualteCost(list_i,list_im1, list_j);
        }
        for (int j = 1; j < m; j++) {
            List list_i = new ArrayList<Double>();
            list_i.add(first.value(0));
            list_i.add(first.value(0));
            list_i.add(first.value(Math.min(step,n-1)));

            List list_j = new ArrayList<Double>();
            list_j.add(second.value(j));
            list_j.add(second.value(Math.max(j-step,0)));
            list_j.add(second.value(Math.min(j+step,m-1)));

            List list_jm1 = new ArrayList<Double>();
            list_jm1.add(second.value(j-1));
            list_jm1.add(second.value(Math.max(j-1-step,0)));
            list_jm1.add(second.value(Math.min(j-1+step,m-1)));
            cost[0][j] = cost[0][j - 1] + calcualteCost(list_j, list_i, list_jm1);
        }

        // Main Loop
        double min;
        for (int i = 1; i < n; i++) {
            min = cutOffValue;
            for (int j = 1; j < m; j++) {
                double d1, d2, d3;

                d1 = cost[i - 1][j - 1] + Math.sqrt((Math.pow(first.value(i) - second.value(j),2)+
                        Math.pow(first.value(Math.max(i-step,0))-second.value(Math.max(j-step,0)),2)+
                        Math.pow(first.value(Math.min(i+step,n-1))-second.value(Math.min(j+step,m-1)),2))/3
                );
                List list_i = new ArrayList<Double>();
                list_i.add(first.value(i));
                list_i.add(first.value(Math.max(i-step,0)));
                list_i.add(first.value(Math.min(i+step,n-1)));

                List list_j = new ArrayList<Double>();
                list_j.add(second.value(j));
                list_j.add(second.value(Math.max(j-step,0)));
                list_j.add(second.value(Math.min(j+step,m-1)));

                List list_im1 = new ArrayList<Double>();
                list_im1.add(first.value(i-1));
                list_im1.add(first.value(Math.max(i-step-1,0)));
                list_im1.add(first.value(Math.min(i+step-1,n-1)));

                List list_jm1 = new ArrayList<Double>();
                list_jm1.add(second.value(j-1));
                list_jm1.add(second.value(Math.max(j-1-step,0)));
                list_jm1.add(second.value(Math.min(j-1+step,m-1)));

                d2 = cost[i - 1][j] + calcualteCost(list_i, list_im1, list_j);
                d3 = cost[i][j - 1] + calcualteCost(list_j, list_i, list_jm1);
                cost[i][j] = Math.min(d1, Math.min(d2, d3));

                if(cost[i][j] >=cutOffValue){
                    cost[i][j] = Double.MAX_VALUE;
                }

                if(cost[i][j] < min){
                    min = cost[i][j];
                }
            }
//            if(min >= cutOffValue){
//                return Double.MAX_VALUE;
//            }
        }
        // Output
//        return cost[m - 1][n - 1];
        return new Pair<>(cost[n-1][m-1],cost);
    }



    public double calcualteCost(List<Double> new_point, List<Double> x, List<Double> y) {

        double dist = 0;
        double dist_new_x = Math.sqrt((Math.pow(new_point.get(0)-x.get(0),2)+
                                      Math.pow(new_point.get(1)-x.get(1),2)+
                                      Math.pow(new_point.get(2)-x.get(2),2))/3
                                        );
        double dist_new_y = Math.sqrt((Math.pow(new_point.get(0)-y.get(0),2)+
                Math.pow(new_point.get(1)-y.get(1),2)+
                Math.pow(new_point.get(2)-y.get(2),2))/3
        );
        double dist_x_y = Math.sqrt((Math.pow(x.get(0)-y.get(0),2)+
                Math.pow(x.get(1)-y.get(1),2)+
                Math.pow(x.get(2)-y.get(2),2))/3
        );
        if (dist_x_y>dist_new_x&&dist_x_y>dist_new_y) {
            dist = c;
        } else {
            dist = c + Math.min(dist_new_x, dist_new_y);
        }

        return dist;
    }


    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void runComparison() throws Exception{
        String tscProbDir = "D:\\datasets\\Univariate_arff\\";
        
//        String datasetName = "ItalyPowerDemand";
//        String datasetName = "GunPoint";
//        String datasetName = "Beef";
//        String datasetName = "Coffee";
        String datasetName = "GunPoint";
//        String datasetName = "BME";

        double c = 0.1;
        Instances train = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TEST");
        
        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        MSMDistance msmOld = new MSMDistance(c);
        knn.setDistanceFunction(msmOld);
        knn.buildClassifier(train);
        
        // new version
        CMSMTraffic1NN msmNew = new CMSMTraffic1NN(c,0.05);
        msmNew.buildClassifier(train);
        
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
            pred = msmNew.classifyInstance(test.instance(i));
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
     * validate for c 5 times
     */
    public static void UCR2018(){
        boolean cv = true;
        Efficient1NNTraffic dtwNew = new CMSMTraffic1NN();
        dtwNew.cparamId = 5;
        dtwNew.UCR2018Cls(cv);
    }

    /**
     * save match path of all test case
     */
    public static void UCR2018SavePathAll(){
        boolean cv = true;
        Efficient1NNTraffic dtwNew = new CMSMTraffic1NN();
        dtwNew .cparamId = 5;
        boolean resample100 =false;
        dtwNew.UCR2018ClsSavePathAll(cv, resample100);
    }

    public static void main(String[] args) throws Exception{
//        
        for(int i = 0; i < 1; i++){
            ///runComparison();
            UCR2018();
//            UCR2018SavePathAll();
        }
    }

    @Override
    public void setParamsFromParamId(Instances train, int paramId) {
        this.c = msmParams[paramId*100/this.cparamId];
        System.out.println(paramId+","+this.c);
    }

    @Override
    public String getParamInformationString() {
        return this.c+","+this.step;
    }


}
