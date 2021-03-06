/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka.classifiers;

import multivariate_timeseriesweka.measures.DTW_D;
import static utilities.InstanceTools.findMinDistance;
import utilities.generic_storage.Pair;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class NN_DTW_D extends AbstractClassifier{
    
    Instances train;
    DTW_D D;
    public NN_DTW_D(){
        D = new DTW_D();
    }
    
    public void setR(double r){
        D.setR(r);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        train = data;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception{
        Pair<Instance, Double> minD = findMinDistance(train, instance, D);
        return minD.var1.classValue();
    }
    
    
}
