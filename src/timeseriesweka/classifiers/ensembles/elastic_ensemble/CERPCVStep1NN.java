// adjusted April '16
// note: not using DTW class in here (redoing the method) as even though the DTW class is already about as efficient, it still
// involves some array copying. Here we can opperate straight on the Instance values instead

package timeseriesweka.classifiers.ensembles.elastic_ensemble;

import timeseriesweka.elastic_distance_measures.ERPDistance;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
//import efficient_standalone_classifiers.Eff

/**
 * @author sjx07ngu
 * validation for step and g, bot bandSize
 */
public class CERPCVStep1NN extends Efficient1NN{

    private double g;
    private double bandSize;
    private double step = 0.05;

    private double[] gValues;
    private double[] windowSizes;
    private boolean gAndWindowsRefreshed = false;

    public CERPCVStep1NN(double g, double bandSize) {
        this.g = g;
        this.bandSize = bandSize;
        this.gAndWindowsRefreshed = false;
        this.classifierIdentifier = "ERP_1NN";
        this.allowLoocv = false;
    }


    public CERPCVStep1NN() {
        // note: default params probably won't suit the majority of problems. Should set through cv or prior knowledge
        this.g = 0.5;
        this.bandSize = 5;
        this.gAndWindowsRefreshed = false;
        this.classifierIdentifier = "ERP_1NN";
    }

    @Override
    public void buildClassifier(Instances train) throws Exception {
        super.buildClassifier(train); 
        this.gAndWindowsRefreshed = false;
        
    }
    
    
    @SuppressWarnings("all")
    public final double distance(Instance first, Instance second, double cutoff){
        
        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
        if (first.classIndex() != first.numAttributes() - 1 || second.classIndex() != second.numAttributes() - 1) {
            return new ERPDistance(this.g, this.bandSize).distance(first, second, cutoff);
        }

        int m = first.numAttributes() - 1;
        int n = second.numAttributes() - 1;
        //int step =(m+n)/40;
        int step =(int)((m+n)/2*this.step);

        // Current and previous columns of the matrix
        double[] curr = new double[m];
        double[] prev = new double[m];

        // size of edit distance band
        // bandsize is the maximum allowed distance to the diagonal
//        int band = (int) Math.ceil(v2.getDimensionality() * bandSize);
        int band = (int) Math.ceil(m * bandSize);

        // g parameter for local usage
        double gValue = g;

        for (int i = 0; i < m; i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            {
                double[] temp = prev;
                prev = curr;
                curr = temp;
            }
            int l = i - (band + 1);
            if (l < 0) {
                l = 0;
            }
            int r = i + (band + 1);
            if (r > (m-1)) {
                r = (m-1);
            }

            for (int j = l; j <= r; j++) {
                if (Math.abs(i - j) <= band) {
                    // compute squared distance of feature vectors
                    double val1 = first.value(i);
                    double val2 = gValue;
                    double diff = Math.pow((val1 - val2),2)+Math.pow(first.value(Math.max(i-step,0))-
                                  val2,2)+Math.pow(first.value(Math.min(i+step,m-1))-val2,2);
                    final double d1 = Math.sqrt(diff );

                    val1 = gValue;
                    val2 = second.value(j);
                    diff = Math.pow((val1 - val2),2)+Math.pow(val1-
                            second.value(Math.max(j-step,0)),2)+Math.pow(val1-
                            second.value(Math.min(j+step,n-1)),2);;
                    final double d2 = Math.sqrt(diff);

                    val1 = first.value(i);
                    val2 = second.value(j);
                    diff = Math.pow((val1 - val2),2)+Math.pow(first.value(Math.max(i-step,0))-
                            second.value(Math.max(j-step,0)),2)+Math.pow(first.value(Math.min(i+step,m-1))-
                            second.value(Math.min(j+step,n-1)),2);
                    final double d12 = Math.sqrt(diff);

                    final double dist1 = d1 * d1;
                    final double dist2 = d2 * d2;
                    final double dist12 = d12 * d12;

                    final double cost;

                    if ((i + j) != 0) {
                        if ((i == 0) || ((j != 0) && (((prev[j - 1] + dist12) > (curr[j - 1] + dist2)) && ((curr[j - 1] + dist2) < (prev[j] + dist1))))) {
                            // del
                            cost = curr[j - 1] + dist2;
                        } else if ((j == 0) || ((i != 0) && (((prev[j - 1] + dist12) > (prev[j] + dist1)) && ((prev[j] + dist1) < (curr[j - 1] + dist2))))) {
                            // ins
                            cost = prev[j] + dist1;
                        } else {
                            // match
                            cost = prev[j - 1] + dist12;
                        }
                    } else {
                        cost = 0;
                    }

                    curr[j] = cost;
                    // steps[i][j] = step;
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
        }

        return Math.sqrt(curr[m-1]);
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
        String datasetName = "UMD";

        double r = 0.1;
        Instances train = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TEST");
        
        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        ERPDistance oldDtw = new ERPDistance(0.1, 0.1);
//        oldDtw.setR(r);
        knn.setDistanceFunction(oldDtw);
        knn.buildClassifier(train);
        
        // new version
        CERPCVStep1NN dtwNew = new CERPCVStep1NN(0.1, 0.1);
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
     * validation for g and step
     */
    public static void UCR2018(){
        double r = 1;
        boolean cv = true;
        Efficient1NN dtwNew = new CERPCVStep1NN(0.1, 1);
        //this.cparamId = 10;
        dtwNew.allowLoocv = cv;
        dtwNew.UCR2018Cls(cv);
    }

    public static void main(String[] args) throws Exception{
        for(int i = 0; i < 1; i++){
            //runComparison();
            UCR2018();
        }
    }

    @Override
    @SuppressWarnings("all")
    public void setParamsFromParamId(Instances train, int paramId) {
        if(!this.gAndWindowsRefreshed){
            double stdv = ERPDistance.stdv_p(train);
            windowSizes = ERPDistance.getInclusive10(0, 0.25);
            gValues = ERPDistance.getInclusive10(0.2*stdv, stdv);
            this.gAndWindowsRefreshed = true;
        }
//        this.g = gValues[paramId/100];
//        this.bandSize = windowSizes[paramId%100/10];
//        this.step = 0.01*paramId%10;

        this.g = gValues[paramId/10];
        this.step = 0.01*paramId%10;
    }

    @Override
    public String getParamInformationString() {
        return this.g+","+this.bandSize+","+this.step;
    }

}
