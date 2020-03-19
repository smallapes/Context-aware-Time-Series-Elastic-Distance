// checked April '16

package timeseriesweka.classifiers.ensembles.elastic_ensemble;


import javafx.util.Pair;
import timeseriesweka.elastic_distance_measures.MSMDistance;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author sjx07ngu
 * @author  use no ArrayList in  cmsm
 */
//public class MSM1NN implements Classifier{
@SuppressWarnings("all")
public class CMSM1NNFast100 extends Efficient1NN{

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

    public CMSM1NNFast100(){
        this.c = 0.1;
        this.step = 0.05;
        this.cparamId = 100;
        this.classifierIdentifier = "CMSM_1NN";
    }

    public CMSM1NNFast100(double c, double step){
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
        
        int m, n;
        if(first.classIndex()==first.numAttributes()-1 && second.classIndex()==second.numAttributes()-1){
            m = first.numAttributes()-1;
            n = second.numAttributes()-1;
        }else{            
            // default case, use the original MSM class (horrible efficiency, but just in as a fail safe for edge-cases) 
            System.err.println("Warning: class designed to use problems with class index as last attribute. Defaulting to original MSM distance");
            MSMDistance msm = new MSMDistance(this.c);
            return new MSMDistance(this.c).distance(first, second);
        }
        int step = (int)((m+n)/2*this.step);

        double[][] cost = new double[m][n];

        // Initialization
        cost[0][0] = Math.abs(first.value(0) - second.value(0));
        double new_point_0,new_point_1,new_point_2,
                x_0,x_1,x_2,y_0,y_1,y_2;
        for (int i = 1; i < m; i++) {
            new_point_0 = first.value(i);
            new_point_1 = first.value(Math.max(i-step,0));
            new_point_2 = first.value(Math.min(i+step,m-1));

            x_0 = second.value(0);
            x_1 = second.value(0);
            x_2 = second.value(Math.min(step,n-1));

            y_0 = first.value(i-1);
            y_1 = first.value(Math.max(i-1-step,0));
            y_2 = first.value(Math.min(i-1+step,m-1));

            cost[i][0] = cost[i - 1][0] + calcualteCost( new_point_0,new_point_1,new_point_2,
                    x_0,x_1,x_2,y_0,y_1,y_2);
        }
        for (int j = 1; j < n; j++) {
            // list_j, list_i, list_jm1
            x_0 = first.value(0);
            x_1 = first.value(0);
            x_2 = first.value(Math.min(step,m-1));

            new_point_0 = second.value(j);
            new_point_1 = second.value(Math.max(j-step,0));
            new_point_2 = second.value(Math.min(j+step,n-1));

            y_0 = second.value(j-1);
            y_1 = second.value(Math.max(j-1-step,0));
            y_2 = second.value(Math.min(j-1+step,n-1));
            cost[0][j] = cost[0][j - 1] + calcualteCost(new_point_0,new_point_1,new_point_2,
                    x_0,x_1,x_2,y_0,y_1,y_2);
        }

        // Main Loop
        double min;
        for (int i = 1; i < m; i++) {
            min = cutOffValue;
            for (int j = 1; j < n; j++) {
                double d1, d2, d3;

                d1 = cost[i - 1][j - 1] + Math.sqrt((Math.pow(first.value(i) - second.value(j),2)+
                                                    Math.pow(first.value(Math.max(i-step,0))-second.value(Math.max(j-step,0)),2)+
                                                    Math.pow(first.value(Math.min(i+step,m-1))-second.value(Math.min(j+step,n-1)),2))/3
                                                    );
                //list_i, list_im1, list_j
                new_point_0 = first.value(i);
                new_point_1 = first.value(Math.max(i-step,0));
                new_point_2 = first.value(Math.min(i+step,m-1));

                y_0 = second.value(j);
                y_1 = second.value(Math.max(j-step,0));
                y_2 = second.value(Math.min(j+step,n-1));

                x_0 = first.value(i-1);
                x_1 = first.value(Math.max(i-step-1,0));
                x_2 = first.value(Math.min(i+step-1,m-1));

                d2 = cost[i - 1][j] + calcualteCost(new_point_0,new_point_1,new_point_2,
                        x_0,x_1,x_2,y_0,y_1,y_2);

                //list_j, list_i, list_jm1
                y_0 = second.value(j-1);
                y_1 = second.value(Math.max(j-1-step,0));
                y_2 = second.value(Math.min(j-1+step,n-1));

                x_0 = first.value(i);
                x_1 = first.value(Math.max(i-step,0));
                x_2 = first.value(Math.min(i+step,m-1));

                new_point_0 = second.value(j);
                new_point_1 = second.value(Math.max(j-step,0));
                new_point_2 = second.value(Math.min(j+step,n-1));

                d3 = cost[i][j - 1] + calcualteCost(new_point_0,new_point_1,new_point_2,
                        x_0,x_1,x_2,y_0,y_1,y_2);
                cost[i][j] = Math.min(d1, Math.min(d2, d3));
                
                if(cost[i][j] >=cutOffValue){
                    cost[i][j] = Double.MAX_VALUE;
                }
                
                if(cost[i][j] < min){
                    min = cost[i][j];
                }
            }
            if(min >= cutOffValue){
                return Double.MAX_VALUE;
            }
        }
        // Output
        return cost[m - 1][n - 1];
    }

    @SuppressWarnings("all")
    public final Pair<Double,double[][]> distanceSavePath(Instance first, Instance second, double cutOffValue) {

        // need to remove class index/ignore
        // simple check - if its last, ignore it. If it's not last, copy the instances, remove that attribue, and then call again

        // Not particularly efficient in the latter case, but a reasonable assumption to make here since all of the UCR/UEA problems
        // match that format.

        int m, n;
//        if(first.classIndex()==first.numAttributes()-1 && second.classIndex()==second.numAttributes()-1){
            m = first.numAttributes()-1;
            n = second.numAttributes()-1;
//        }else{
//            // default case, use the original MSM class (horrible efficiency, but just in as a fail safe for edge-cases)
//            System.err.println("Warning: class designed to use problems with class index as last attribute. Defaulting to original MSM distance");
//            MSMDistance msm = new MSMDistance(this.c);
//            return new MSMDistance(this.c).distance(first, second);
//        }
        int step = (int)((m+n)/2*this.step);

        double[][] cost = new double[m][n];

        // Initialization
        cost[0][0] = Math.abs(first.value(0) - second.value(0));
        double new_point_0,new_point_1,new_point_2;
        double x_0,x_1,x_2;
        double y_0,y_1,y_2;

        for (int i = 1; i < m; i++) {
            new_point_0 = first.value(i);
            new_point_1 = first.value(Math.max(i-step,0));
            new_point_2 = first.value(Math.min(i+step,m-1));

            x_0 = second.value(0);
            x_1 = second.value(0);
            x_2 = second.value(Math.min(step,n-1));

            y_0 = first.value(i-1);
            y_1 = first.value(Math.max(i-1-step,0));
            y_2 = first.value(Math.min(i-1+step,m-1));

            cost[i][0] = cost[i - 1][0] + calcualteCost( new_point_0,new_point_1,new_point_2,
                    x_0,x_1,x_2,y_0,y_1,y_2);
        }
        for (int j = 1; j < n; j++) {

            // list_j, list_i, list_jm1
            x_0 = first.value(0);
            x_1 = first.value(0);
            x_2 = first.value(Math.min(step,m-1));

            new_point_0 = second.value(j);
            new_point_1 = second.value(Math.max(j-step,0));
            new_point_2 = second.value(Math.min(j+step,n-1));

            y_0 = second.value(j-1);
            y_1 = second.value(Math.max(j-1-step,0));
            y_2 = second.value(Math.min(j-1+step,n-1));
            cost[0][j] = cost[0][j - 1] + calcualteCost(new_point_0,new_point_1,new_point_2,
                    x_0,x_1,x_2,y_0,y_1,y_2);
        }

        // Main Loop
        double min;
        for (int i = 1; i < m; i++) {
            min = cutOffValue;
            for (int j = 1; j < n; j++) {
                double d1, d2, d3;

                d1 = cost[i - 1][j - 1] + Math.sqrt((Math.pow(first.value(i) - second.value(j),2)+
                        Math.pow(first.value(Math.max(i-step,0))-second.value(Math.max(j-step,0)),2)+
                        Math.pow(first.value(Math.min(i+step,m-1))-second.value(Math.min(j+step,n-1)),2))/3
                );

                //list_i, list_im1, list_j
                new_point_0 = first.value(i);
                new_point_1 = first.value(Math.max(i-step,0));
                new_point_2 = first.value(Math.min(i+step,m-1));

                y_0 = second.value(j);
                y_1 = second.value(Math.max(j-step,0));
                y_2 = second.value(Math.min(j+step,n-1));

                x_0 = first.value(i-1);
                x_1 = first.value(Math.max(i-step-1,0));
                x_2 = first.value(Math.min(i+step-1,m-1));

                d2 = cost[i - 1][j] + calcualteCost(new_point_0,new_point_1,new_point_2,
                        x_0,x_1,x_2,y_0,y_1,y_2);

                //list_j, list_i, list_jm1
                y_0 = second.value(j-1);
                y_1 = second.value(Math.max(j-1-step,0));
                y_2 = second.value(Math.min(j-1+step,n-1));

                x_0 = first.value(i);
                x_1 = first.value(Math.max(i-step,0));
                x_2 = first.value(Math.min(i+step,m-1));

                new_point_0 = second.value(j);
                new_point_1 = second.value(Math.max(j-step,0));
                new_point_2 = second.value(Math.min(j+step,n-1));

                d3 = cost[i][j - 1] + calcualteCost(new_point_0,new_point_1,new_point_2,
                        x_0,x_1,x_2,y_0,y_1,y_2);
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
//        return cost[m - 1][n - 1];
        return new Pair<>(cost[m-1][n-1],cost);
    }


    public double calcualteCost(double new_point_0,double new_point_1,double new_point_2,
                                double x_0, double x_1, double x_2,
                                double y_0, double y_1, double y_2) {
        ;
        double dist = 0;
        double dist_new_x = Math.sqrt((Math.pow(new_point_0-x_0,2)+
                Math.pow(new_point_1-x_1,2)+
                Math.pow(new_point_2-x_2,2))/3
        );
        double dist_new_y = Math.sqrt((Math.pow(new_point_0-y_0,2)+
                Math.pow(new_point_1-y_1,2)+
                Math.pow(new_point_2-y_2,2))/3
        );
        double dist_x_y = Math.sqrt((Math.pow(x_0-y_0,2)+
                Math.pow(x_1-y_1,2)+
                Math.pow(x_2-y_2,2))/3
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
        CMSM1NNFast100 msmNew = new CMSM1NNFast100(c,0.05);
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
        Efficient1NN dtwNew = new CMSM1NNFast100();
        dtwNew.cparamId = 100;
        dtwNew.UCR2018Cls(cv);
    }

    /**
     * Author :tcl
     * validate for c 5 times
     */
    public static void UCR2018Time(){
        boolean cv = true;
        Efficient1NN dtwNew = new CMSM1NNFast100();
        dtwNew.cparamId = 100;
        dtwNew.saveTrainTime = true;
        dtwNew.UCR2018Cls(cv);
    }

    /**
     * save match path of all test case
     */
    public static void UCR2018SavePathAll(){
        boolean cv = true;
        Efficient1NN dtwNew = new CMSM1NNFast100();
        dtwNew .cparamId = 5;
        boolean resample100 =false;
        dtwNew.UCR2018ClsSavePathAll(cv, resample100);
    }

    public static void main(String[] args) throws Exception{
//        
        for(int i = 0; i < 1; i++){
            ///runComparison();
            UCR2018();
//            UCR2018Time();
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
