/**
 *
 * @author ajb
 *local class to run experiments with the UCR-UEA or UCI data


*/
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import multivariate_timeseriesweka.classifiers.*;
import timeseriesweka.classifiers.BOSS;
import timeseriesweka.classifiers.BagOfPatterns;
import timeseriesweka.classifiers.DD_DTW;
import timeseriesweka.classifiers.DTD_C;
import timeseriesweka.classifiers.ElasticEnsemble;
import timeseriesweka.classifiers.FastShapelets;
import timeseriesweka.classifiers.FlatCote;
import timeseriesweka.classifiers.HiveCote;
import timeseriesweka.classifiers.LPS;
import timeseriesweka.classifiers.LearnShapelets;
import timeseriesweka.classifiers.NN_CID;
import timeseriesweka.classifiers.ParameterSplittable;
import timeseriesweka.classifiers.RISE;
import timeseriesweka.classifiers.SAXVSM;
import timeseriesweka.classifiers.ShapeletTransformClassifier;
import timeseriesweka.classifiers.SlowDTW_1NN;
import timeseriesweka.classifiers.TSBF;
import timeseriesweka.classifiers.TSF;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.ED1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.MSM1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.WDTW1NN;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import vector_classifiers.TunedSVM;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.TunedRotationForest;
import utilities.ClassifierResults;
import vector_classifiers.CAWPE;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import timeseriesweka.classifiers.FastWWS.FastDTWWrapper;
import utilities.InstanceTools;
import utilities.multivariate_tools.MultivariateInstanceTools;
import vector_classifiers.*;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instances;

public class ExperimentsKaggle{
//For threaded version
    String[] args;
    public static int folds=30; 
    static int numCVFolds = 10;
    static boolean debug=true;
    static boolean checkpoint=false;
    static boolean generateTrainFiles=true;
    static Integer parameterNum=0;
    static boolean singleFile=false;
    static boolean foldsInFile=false;
    static boolean useBagsSampling=false;//todo is a hack for bags project experiments 
    static double SPLITPROP=0.5; 
    
    static boolean predictionOutput = false;
    static boolean includeLastClass = false;
    static double threshhold = 0.5;

/** This method is now too bloated
 * 
 * @param classifier
 * @param fold
 * @return 
 */    
    public static Classifier setClassifier(String classifier, int fold){
        Classifier c=null;
        TunedSVM svm=null;
        switch(classifier){
            case "ContractRotationForest":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setDayLimit(5);
                ((ContractRotationForest)c).setSeed(fold);
                
                break;
            case "ContractRotationForest1Day":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(24);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest5Minutes":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setMinuteLimit(5);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest30Minutes":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setMinuteLimit(30);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest1Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(1);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest2Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(2);
                break;
            case "ContractRotationForest3Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(3);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest12Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(12);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            
            case "ShapeletI": case "Shapelet_I": case "ShapeletD": case "Shapelet_D": case  "Shapelet_Indep"://Multivariate version 1
                c=new MultivariateShapeletTransformClassifier();
//Default to 1 day max run: could do this better
                ((MultivariateShapeletTransformClassifier)c).setOneDayLimit();
                ((MultivariateShapeletTransformClassifier)c).setSeed(fold);
                ((MultivariateShapeletTransformClassifier)c).setTransformType(classifier);
                break;
            case "ED_I":
                c=new NN_ED_I();
                break;
            case "ED_D":
                c=new NN_ED_D();
                break;
            case "DTW_I":
                c=new NN_DTW_I();
                break;
            case "DTW_D":
                c=new NN_DTW_D();
                break;
            case "DTW_A":
                c=new NN_DTW_A();
                break;
//TIME DOMAIN CLASSIFIERS   
            
            case "ED":
                c=new ED1NN();
                break;
            case "C45":
                c=new J48();
                break;
            case "NB":
                c=new NaiveBayes();
                break;
            case "SVML":
                c=new SMO();
                PolyKernel p=new PolyKernel();
                p.setExponent(1);
                ((SMO)c).setKernel(p);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);
                break;
            case "SVMQ": case "SVMQuad":
                c=new SMO();
                PolyKernel poly=new PolyKernel();
                poly.setExponent(2);
                ((SMO)c).setKernel(poly);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);

                
                /*                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.QUADRATIC);
                svm.optimiseParas(false);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
 */               break;
            case "SVMRBF": 
                c=new SMO();
                RBFKernel rbf=new RBFKernel();
                rbf.setGamma(0.5);
                ((SMO)c).setC(5);
                ((SMO)c).setKernel(rbf);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);
                
                break;
            case "BN":
                c=new BayesNet();
                break;
            case "MLP":
                c=new MultilayerPerceptron();
                break;
            case "TwoLayerMLP":
                TunedTwoLayerMLP twolayer=new TunedTwoLayerMLP();
                twolayer.setParamSearch(false);
                twolayer.setSeed(fold);
                c= twolayer;
                break;
                
            case "RandFOOB":
                c= new TunedRandomForest();
                ((RandomForest)c).setNumTrees(500);
                ((TunedRandomForest)c).tuneParameters(false);
                ((TunedRandomForest)c).setCrossValidate(false);
                ((TunedRandomForest)c).setEstimateAcc(true);
                ((TunedRandomForest)c).setSeed(fold);
                ((TunedRandomForest)c).setDebug(debug);
                
                break;
            case "RandF": case "RandomForest": case "RandF500": case "RandomForest500":
                c= new TunedRandomForest();
                ((RandomForest)c).setNumTrees(500);
                ((TunedRandomForest)c).tuneParameters(false);
                ((TunedRandomForest)c).setCrossValidate(true);
                ((TunedRandomForest)c).setSeed(fold);
                break;
            case "RandF10000":
                c= new TunedRandomForest();
                ((RandomForest)c).setNumTrees(10000);
                ((TunedRandomForest)c).tuneParameters(false);
                ((TunedRandomForest)c).setCrossValidate(false);
                ((TunedRandomForest)c).setSeed(fold);
                break;


            case "RotF": case "RotationForest": case "RotF200": case "RotationForest200":
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(200);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFRandomTree":
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(200);
                ((RotationForest)c).setClassifier(new RandomTree());
                
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;


            case "RotFBootstrap":
                c= new RotationForestBootstrap();
                ((RotationForestBootstrap)c).setNumIterations(200);
                ((RotationForestBootstrap)c).setSeed(fold);
                ((RotationForestBootstrap)c).tuneParameters(false);
                ((RotationForestBootstrap)c).setSeed(fold);
                ((RotationForestBootstrap)c).estimateAccFromTrain(false);
                break;
            case "RotFLimited":
                c= new RotationForestLimitedAttributes();
                ((RotationForestLimitedAttributes)c).setNumIterations(200);
                ((RotationForestLimitedAttributes)c).tuneParameters(false);
                ((RotationForestLimitedAttributes)c).setSeed(fold);
                ((RotationForestLimitedAttributes)c).estimateAccFromTrain(false);
                break;
            case "TunedRandF":
                c= new TunedRandomForest();
                ((TunedRandomForest)c).tuneParameters(true);
                ((TunedRandomForest)c).setCrossValidate(true);
                ((TunedRandomForest)c).setSeed(fold);             
                ((TunedRandomForest)c).setDebug(debug);
                break;
            case "TunedRandFOOB":
                c= new TunedRandomForest();
                ((TunedRandomForest)c).tuneParameters(true);
                ((TunedRandomForest)c).setCrossValidate(false);
                ((TunedRotationForest)c).setSeed(fold);
                break;
            case "TunedRotF":
                c= new TunedRotationForest();
                ((TunedRotationForest)c).tuneParameters(true);
                ((TunedRotationForest)c).setSeed(fold);
                break;
            case "TunedSVMRBF":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.RBF);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
                break;
            case "TunedSVMQuad":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.QUADRATIC);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                svm.setLargePolynomialParameterSpace(1089);                
                c= svm;
                break;
            case "TunedSVMLinear":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.LINEAR);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                svm.setLargePolynomialParameterSpace(1089);
                c= svm;
                break;
            case "TunedSVMPolynomial":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.POLYNOMIAL);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
                break;
            case "TunedSVMKernel":
                svm=new TunedSVM();
                svm.optimiseParas(true);
                svm.optimiseKernel(true);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
                break;
            case "TunedSingleLayerMLP":
                TunedSingleLayerMLP mlp=new TunedSingleLayerMLP();
                mlp.setParamSearch(true);
                mlp.setTrainingTime(200);
                mlp.setSeed(fold);
                c= mlp;
                break;
            case "TunedTwoLayerMLP":
                TunedTwoLayerMLP mlp2=new TunedTwoLayerMLP();
                mlp2.setParamSearch(true);
                mlp2.setSeed(fold);
                c= mlp2;
                break;
            case "TunedMultiLayerPerceptron":
                TunedMultiLayerPerceptron mlp3=new TunedMultiLayerPerceptron();
               
                mlp3.setParamSearch(true);
                mlp3.setSeed(fold);
                mlp3.setTrainingTime(200);
                c= mlp3;
                break;
            case "RandomRotationForest1":
                c= new RotationForestLimitedAttributes();
                ((RotationForestLimitedAttributes)c).setNumIterations(200);
                ((RotationForestLimitedAttributes)c).setMaxNumAttributes(100);
                break;
            case "Logistic":
                c= new Logistic();
                break;
            case "NN":
                kNN k=new kNN(100);
                k.setCrossValidate(true);
                k.normalise(false);
                k.setDistanceFunction(new EuclideanDistance());
                return k;
            case "HESCA":
            case "CAWPE":
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);
                break;
            case "CAWPEPLUS":
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);                
                ((CAWPE)c).setAdvancedCAWPESettings();
                break;
            case "CAWPEFROMFILE":
                String[] classifiers={"XGBoost","RandF","RotF"};
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);  
                ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                ((CAWPE)c).setResultsFileLocationParameters(horribleGlobalPath, datasetName, fold);
                
                ((CAWPE)c).setClassifiersNamesForFileRead(classifiers);
                
                
                break;
            case "CAWPE_AS_COTE":
                String[] cls={"CAWPEFROMFILE","SLOWDTWCV","ST","TSF"};
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);  
                ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                ((CAWPE)c).setResultsFileLocationParameters(horribleGlobalPath, datasetName, fold);
                ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                break;
            case "XGBoost":
                c=new TunedXGBoost();
                ((TunedXGBoost)c).setTuneParameters(false);
                ((TunedXGBoost)c).setSeed(fold);
                break;

//ELASTIC CLASSIFIERS     
            case "EE": case "ElasticEnsemble":
                c=new ElasticEnsemble();
                break;
            case "DTW":
                c=new DTW1NN();
                ((DTW1NN )c).setWindow(1);
                break;
            case "SLOWDTWCV":
//                c=new DTW1NN();
                c=new SlowDTW_1NN();
                ((SlowDTW_1NN)c).optimiseWindow(true);
                break;
            case "DTWCV":
//                c=new DTW1NN();
//                c=new FastDTW_1NN();
//                ((FastDTW_1NN)c).optimiseWindow(true);
//                break;
//            case "FastDTWWrapper":
                c= new FastDTWWrapper();
                break;
            case "DD_DTW":
                c=new DD_DTW();
                break;
            case "DTD_C":
                c=new DTD_C();
                break;
            case "CID_DTW":
                c=new NN_CID();
                ((NN_CID)c).useDTW();
                break;
            case "MSM":
                c=new MSM1NN();
                break;
            case "TWE":
                c=new MSM1NN();
                break;
            case "WDTW":    
                c=new WDTW1NN();
                break;
                
            case "LearnShapelets": case "LS":
                c=new LearnShapelets();
                break;
            case "FastShapelets": case "FS":
                c=new FastShapelets();
                break;
            case "ShapeletTransform": case "ST": case "ST_Ensemble": case "ShapeletTransformClassifier":
                c=new ShapeletTransformClassifier();
//Default to 1 day max run: could do this better
                ((ShapeletTransformClassifier)c).setOneDayLimit();
                ((ShapeletTransformClassifier)c).setSeed(fold);
                break;
            case "RISE":
                c=new RISE();
                break;
            case "RISEV2":
                c=new RiseV2();
                ((RiseV2)c).buildFromSavedData(true);
                break;
            case "TSBF":
                c=new TSBF();
                break;
            case "BOP": case "BoP": case "BagOfPatterns":
                c=new BagOfPatterns();
                break;
             case "BOSS": case "BOSSEnsemble": 
                c=new BOSS();
                break;
            case "TSF":
                c=new TSF();
                break;
             case "SAXVSM": case "SAX": 
                c=new SAXVSM();
                break;
             case "LPS":
                c=new LPS();
                break; 
             case "FlatCOTE":
                c=new FlatCote();
                break; 
             case "HiveCOTE": case "HIVECOTE": case "HiveCote": case "Hive-COTE":
                c=new HiveCote();
                ((HiveCote)c).setContract(24);
                break; 
            case "TunedXGBoost":
                 c=new TunedXGBoost();
                ((TunedXGBoost)c).setSeed(fold);
                ((TunedXGBoost)c).setDebug(false);
                ((TunedXGBoost)c).setDebugPrinting(false);
                ((TunedXGBoost)c).setTuneParameters(true);
                 break;
            case "RotFDefault":
                c = new RotationForest();
                ((RotationForest)c).setSeed(fold);
                return c;
//Hacky bit for paper
            case "RotF10": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(10);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF50": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(50);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF100": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(100);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF150": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(150);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF250": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(250);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF300": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(300);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF350": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(350);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF400": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(400);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF450": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(450);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF500": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(450);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
//1000 attributes per group (10 values) 3; 4; : : : ; 12g
            case "RotFG3": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(3);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG4": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(4);
                ((RotationForest)c).setMaxGroup(4);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG5": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(5);
                ((RotationForest)c).setMaxGroup(5);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG6": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(6);
                ((RotationForest)c).setMaxGroup(6);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG7": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(7);
                ((RotationForest)c).setMaxGroup(7);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG8": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(8);
                ((RotationForest)c).setMaxGroup(8);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG9": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(9);
                ((RotationForest)c).setMaxGroup(9);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG10": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(10);
                ((RotationForest)c).setMaxGroup(10);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG11": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(11);
                ((RotationForest)c).setMaxGroup(11);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG12": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(12);
                ((RotationForest)c).setMaxGroup(12);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
                
                
//combinations sampling proportion (10 values) f0:1; 0:2; : : : ; 1:0g                
            case "RotRP1": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(0);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP2": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(10);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP3": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(20);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP4": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(30);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP5": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(40);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP6": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(50);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP7": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(60);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP8": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(70);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP9": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(80);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP10": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(90);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
                
           default:
                System.out.println("UNKNOWN CLASSIFIER "+classifier);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;
    }
        
    public static String horribleGlobalPath="";
    public static String datasetName="";
    public static void main(String[] args) throws Exception{

//IF args are passed, it is a cluster run. Otherwise it is a local run, either threaded or not
        debug=false;

//        foldsInFile=true;
        for(String str:args)
            System.out.println(str);
        if(args.length<6){
    //Args 1 and 2 are problem and results path
            String[] newArgs=new String[6];
            newArgs[0]="//cmptscsvr.cmp.uea.ac.uk/ueatsc/BagsSDM/Data/";//All on the beast now
            newArgs[1]="//cmptscsvr.cmp.uea.ac.uk/ueatsc/BagsSDM/Results/";
    //Arg 3 argument is whether to cross validate or not and produce train files
                newArgs[2]="true";
    // Arg 4,5,6 Classifier, Problem, Fold  
                String[] names={"CAWPE_AS_COTE"};
                
            for(String str:names){
                    newArgs[3]=str;
//These are set in the localX method
//              newArgs[4]="Adiac";
//                newArgs[5]="1";
//                String[] problems=DataSets.fileNames;
                    String[] problems=new String[]{"SieveBagsTwoClassHisto"};
                    //"GTtoSieveTwoClassHisto","SieveBagsTwoClassHisto",
                        //"FakeBagsTwoClassHisto","FakeBagsFiveClassHisto"};
                //"BagsTwoClassHisto","BagsFiveClassHisto", "leaveOutOneElectricalItemHisto",
//                "GTtoSieveTwoClassHisto","leaveOutOneElectricalItemHisto","SieveBagsTwoClassHisto"};
                    int folds=18;
                    //threaded=false;
                    horribleGlobalPath="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\BagsSDM\\Results\\";
//                    if(threaded){//Do problems listed threaded 
//                        localThreadedRun(newArgs,problems,folds);
//                    }
//                    else //Do the problems listed sequentially
//                        localSequentialRun(newArgs,problems,folds);
            }
        }        
        else{    
            singleExperiment(args);  
        }
    }
    
    public static void singleExperiment(String[] args) throws Exception{
            DataSets.problemPath=args[0];
            DataSets.resultsPath=args[1];
//Arg 3 argument is whether to cross validate or not and produce train files
            generateTrainFiles=Boolean.parseBoolean(args[2]);
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
                f.setWritable(true, false);
            }
// Arg 4,5,6 Classifier, Problem, Fold             
            String[] newArgs=new String[3];
            for(int i=0;i<3;i++)
                newArgs[i]=args[i+3];
//OPTIONAL
//  Arg 7:  whether to checkpoint        
            checkpoint=false;
            if(args.length>=7){
                String s=args[args.length-1].toLowerCase();
                if(s.equals("true"))
                    checkpoint=true;
            }
//Arg 8: if present, do a single parameter split
            parameterNum=0;
            if(args.length>=8){
                parameterNum=Integer.parseInt(args[7]);
            }       
 //  Arg 9:  outputs predictions in kaggle format       
            predictionOutput=false;
            if(args.length>=9){
                String s=args[8].toLowerCase();
                if(s.equals("true"))
                    predictionOutput=true;
            }
            
  //  Arg 10:  outputs predictions in kaggle format       
            includeLastClass=false;
            if(args.length>=10){
                String s=args[9].toLowerCase();
                if(s.equals("true"))
                    includeLastClass=true;
            }
            
            ExperimentsKaggle.singleClassifierAndFoldTrainTestSplit(newArgs); 
    }
    /** Run a given classifier/problem/fold combination with associated file set up
 @param args: 
 * args[0]: Classifier name. Create classifier with setClassifier
 * args[1]: Problem name
 * args[2]: Fold number. This is assumed to range from 1, hence we subtract 1
 * (this is because of the scripting we use to run the code on the cluster)
 *          the standard archive folds are always fold 0
 * 
 * NOTES: 
 * 1. this assumes you have set DataSets.problemPath to be where ever the 
 * data is, and assumes the data is in its own directory with two files, 
 * args[1]_TRAIN.arff and args[1]_TEST.arff 
 * 2. assumes you have set DataSets.resultsPath to where you want the results to
 * go It will NOT overwrite any existing results (i.e. if a file of non zero 
 * size exists)
 * 3. This method just does the file set up then calls the next method. If you 
 * just want to run the problem, go to the next method
* */
    public static void singleClassifierAndFoldTrainTestSplit(String[] args) throws Exception{
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        String predictions = DataSets.resultsPath+classifier+"/Predictions/"+problem;
        File f=new File(predictions);
        if(!f.exists())
            f.mkdirs();
        
        //Check whether fold already exists, if so, dont do it, just quit
        if(!CollateResults.validateSingleFoldFile(predictions+"/testFold"+fold+".csv"))
        {
            Classifier c=setClassifier(classifier,fold);;
            
            Instances[] data = new Instances[2];
            
            if(!predictionOutput){
                if (MultiVariateProcessing.isMultivariateClassifier(classifier)){
                    File trainFile=new File(DataSets.problemPath+problem+"/"+problem+"_TRAIN.arff");
                    data = InstanceTools.resampleInstances(ClassifierTools.loadData(trainFile.getAbsolutePath()), fold, SPLITPROP);
                }
                else{
                    Instances train = MultiVariateProcessing.convertToUnivariateTrain(DataSets.problemPath, DataSets.problemPath, problem);
                    problem = problem+"_UNI";
                    data = InstanceTools.resampleInstances(train, fold, SPLITPROP);
                }
            }
            else{
                //checks if converts to univariate if classifiers cannot handle multivariate
                if (MultiVariateProcessing.isMultivariateClassifier(classifier)){
                    File trainFile=new File(DataSets.problemPath+problem+"/"+problem+"_TRAIN.arff");
                    File testFile=new File(DataSets.problemPath+problem+"/"+problem+"_TEST.arff");
                    data[0]=ClassifierTools.loadData(trainFile.getAbsolutePath());
                    data[1]=ClassifierTools.loadData(testFile.getAbsolutePath());
                }
                else {
                    data = MultiVariateProcessing.convertToUnivariate(DataSets.problemPath, DataSets.problemPath, problem);
                }
            }

            if(parameterNum>0 && c instanceof ParameterSplittable)//Single parameter fold
            {
                checkpoint=false;
//Check if it already exists, if it does, exit
                if(CollateResults.validateSingleFoldFile(predictions+"/fold"+fold+"_"+parameterNum+".csv")){ //Exit
                    System.out.println("Fold "+predictions+"/fold"+fold+"_"+parameterNum+".csv  already exists");
                    return; //Aready done
                }
            }
            
            double acc = singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
            System.out.println("Classifier="+classifier+", Problem="+problem+", Fold="+fold+", Test Acc,"+acc);
        }
    }
    
/**
 * 
 * @param train: the standard train fold Instances from the archive 
 * @param test: the standard test fold Instances from the archive
 * @param c: Classifier to evaluate
 * @param fold: integer to indicate which fold. Set to 0 to just use train/test
 * @param resultsPath: a string indicating where to store the results
 * @return the accuracy of c on fold for problem given in train/test
 * 
 * NOTES:
 * 1.  If the classifier is a SaveableEnsemble, then we save the internal cross 
 * validation accuracy and the internal test predictions
 * 2. The output of the file testFold+fold+.csv is
 * Line 1: ProblemName,ClassifierName, train/test
 * Line 2: parameter information for final classifier, if it is available
 * Line 3: test accuracy
 * then each line is
 * Actual Class, Predicted Class, Class probabilities 
 * 
 * 
 */    
    public static double singleClassifierAndFoldTrainTestSplit(Instances train, Instances test, Classifier c, int fold,String resultsPath){
        String testFoldPath="/testFold"+fold+".csv";
        String trainFoldPath="/trainFold"+fold+".csv";
        
        ClassifierResults trainResults = null;
        ClassifierResults testResults = null;
        if(parameterNum>0 && c instanceof ParameterSplittable)//Single parameter fold
        {
//If TunedRandForest or TunedRotForest need to let the classifier know the number of attributes 
//n orderto set parameters
            if(c instanceof TunedRandomForest)
                ((TunedRandomForest)c).setNumFeaturesInProblem(train.numAttributes()-1);
            checkpoint=false;
            ((ParameterSplittable)c).setParametersFromIndex(parameterNum);
//            System.out.println("classifier paras =");
            trainFoldPath="/fold"+fold+"_"+parameterNum+".csv";
            generateTrainFiles=true;
        }
        else{
//Only do all this if not an internal fold
    // Save internal info for ensembles
            if(c instanceof SaveableEnsemble)
               ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
            if(checkpoint && c instanceof SaveEachParameter){     
                ((SaveEachParameter) c).setPathToSaveParameters(resultsPath+"/fold"+fold+"_");
            }
        }
        
        try{             
            if(generateTrainFiles){
                if(c instanceof TrainAccuracyEstimate){ //Classifier will perform cv internally
                    ((TrainAccuracyEstimate)c).writeCVTrainToFile(resultsPath+trainFoldPath);
                    File f=new File(resultsPath+trainFoldPath);
                    if(f.exists())
                        f.setWritable(true, false);
                }
                else{ // Need to cross validate here
                    
                    if(c instanceof RiseV2 && ((RiseV2)c).getBuildFromSavedData()){
                        //Write some internal crossvalidation that can deal with read from files.
                    }else{
                        CrossValidator cv = new CrossValidator();
                        cv.setSeed(fold);
                        int numFolds = Math.min(train.numInstances(), numCVFolds);
                        cv.setNumFolds(numFolds);
                        trainResults=cv.crossValidateWithStats(c,train);
                    }    
                }
            }
            
            //Build on the full train data here
            long buildTime=System.currentTimeMillis();
            c.buildClassifier(train);
            buildTime=System.currentTimeMillis()-buildTime;
            
            if (generateTrainFiles) { //And actually write the full train results if needed
                if(!(c instanceof TrainAccuracyEstimate)){ 
                    OutFile trainOut=new OutFile(resultsPath+trainFoldPath);
                    trainOut.writeLine(train.relationName()+","+c.getClass().getName()+",train");
                    if(c instanceof SaveParameterInfo )
                        trainOut.writeLine(((SaveParameterInfo)c).getParameters()); //assumes build time is in it's param info, is for tunedsvm
                    else 
                        trainOut.writeLine("BuildTime,"+buildTime+",No Parameter Info");
                    trainOut.writeLine(trainResults.acc+"");
                    trainOut.writeLine(trainResults.writeInstancePredictions());
                    //not simply calling trainResults.writeResultsFileToString() since it looks like those that extend SaveParameterInfo will store buildtimes
                    //as part of their params, and so would be written twice
                    trainOut.closeFile();
                    File f=new File(resultsPath+trainFoldPath);
                    if(f.exists())
                        f.setWritable(true, false);
                    
                }
            }
            if(parameterNum==0 && predictionOutput)//Not a single parameter fold
            {  
                //Start of testing, only doing this if the test file doesnt exist
                //This is checked before the buildClassifier also, but we have a special case for the file builder
                //that copies the results over in buildClassifier. No harm in checking again!
                if(!CollateResults.validateSingleFoldFile(resultsPath+testFoldPath)){
                    int numInsts = test.numInstances();
                    
                    InFile ids=new InFile(DataSets.problemPath+"ids.txt");
                    OutFile testOut=new OutFile(resultsPath+"predictions.csv");
                    testOut.writeLine(ids.readLine());

                    for(int testInstIndex = 0; testInstIndex < numInsts; testInstIndex++) {
                        test.instance(testInstIndex).setClassMissing();//and remove from each instance given to the classifier (just to be sure)

                        //make prediction
                        double[] probs=c.distributionForInstance(test.instance(testInstIndex));
                        
                        double cls15 = 0;
                        
                        if (includeLastClass){
                            double max = 0;
                            for(int i = 0; i < probs.length; i++){
                                if (probs[i] > max){
                                    max = probs[i];
                                }
                            }
                            
                            //method 1
                            if (max < threshhold){
                                probs = new double[14];
                                cls15 = 1;
                            }
                            
                            //method 2
//                            double diff = 100 - max;
//                            if (diff > 20){
//                                for(int i = 0; i < probs.length; i++){
//                                    double tax = (probs[i]*(diff/100.0f));
//                                    probs[i] -= tax;
//                                    cls15 += tax;
//                                }
//                            }
                        }
                        
                        StringBuilder sb = new StringBuilder(ids.readLine());
                        for(int i = 0; i < probs.length; i++){
                            sb.append(",").append(Double.toString(probs[i]));
                        }
                        sb.append(",").append(cls15);
                        
                        testOut.writeLine(sb.toString());
                    }

                    testOut.closeFile();
                    File f=new File(resultsPath+testFoldPath);
                    if(f.exists())
                        f.setWritable(true, false);
                    
                }
                return 0;
            }
            else if(parameterNum==0)//Not a single parameter fold
            {  
                //Start of testing, only doing this if the test file doesnt exist
                //This is checked before the buildClassifier also, but we have a special case for the file builder
                //that copies the results over in buildClassifier. No harm in checking again!
                if(!CollateResults.validateSingleFoldFile(resultsPath+testFoldPath)){
                    int numInsts = test.numInstances();
                    int pred;
                    testResults = new ClassifierResults(test.numClasses());
                    double[] trueClassValues = test.attributeToDoubleArray(test.classIndex()); //store class values here

                    for(int testInstIndex = 0; testInstIndex < numInsts; testInstIndex++) {
                        test.instance(testInstIndex).setClassMissing();//and remove from each instance given to the classifier (just to be sure)

                        //make prediction
                        double[] probs=c.distributionForInstance(test.instance(testInstIndex));
                        testResults.storeSingleResult(probs);
                    }
                    testResults.finaliseResults(trueClassValues); 

                    //Write results
                    OutFile testOut=new OutFile(resultsPath+testFoldPath);
                    testOut.writeLine(test.relationName()+","+c.getClass().getName()+",test");
                    if(c instanceof SaveParameterInfo)
                      testOut.writeLine(((SaveParameterInfo)c).getParameters());
                    else
                        testOut.writeLine("No parameter info");
                    testOut.writeLine(testResults.acc+"");
                    testOut.writeString(testResults.writeInstancePredictions());
                    testOut.closeFile();
                    File f=new File(resultsPath+testFoldPath);
                    if(f.exists())
                        f.setWritable(true, false);
                    
                }
                return testResults.acc;
            }
            else
                 return 0;//trainResults.acc;   
        } catch(Exception e) {
            System.out.println(" Error ="+e+" in method simpleExperiment");
            e.printStackTrace();
            System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
            System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes and "+test.numInstances()+" instances");
            System.out.println(" Classifier ="+c.getClass().getName()+" fold = "+fold);
            System.out.println(" Results path is "+ resultsPath);
                    
            return Double.NaN;
        }
    }     
}