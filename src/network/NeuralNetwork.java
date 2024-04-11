package network;

import data.Image;
import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;

import java.io.*;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Scanner;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class NeuralNetwork {

    List<Layer> _layers;
    double scaleFactor;

    public NeuralNetwork(List<Layer> _layers, double scaleFactor) {
        this._layers = _layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers(){

        if(_layers.size() <= 1){
            return;
        }

        for(int i = 0; i < _layers.size(); i++){
            if(i == 0){
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            } else if (i == _layers.size()-1){
                _layers.get(i).set_previousLayer(_layers.get(i-1));
            } else {
                _layers.get(i).set_previousLayer(_layers.get(i-1));
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            }
        }
    }

    public double[] getErrors(double[] networkOutput, int correctAnswer){
        int numClasses = networkOutput.length;

        double[] expected = new double[numClasses];

        expected[correctAnswer] = 1;

        return add(networkOutput, multiply(expected, -1));
    }

    private int getMaxIndex(double[] in){

        double max = 0;
        int index = 0;

        for(int i = 0; i < in.length; i++){
            if(in[i] >= max){
                max = in[i];
                index = i;
            }

        }

        return index;
    }

    public int guess(Image image){
        List<double[][]> inList = new ArrayList<>();
        inList.add(multiply(image.getData(), (1.0/scaleFactor)));

        double[] out = _layers.get(0).getOutput(inList);
        int guess = getMaxIndex(out);

        //System.out.println(_layers.get(0));

        return guess;
    }

    public float test (List<Image> images){
        int correct = 0;

        for(Image img: images){
            int guess = guess(img);

            if(guess == img.getLabel()){
                correct++;
            }
        }

        return((float)correct/images.size());
    }

    public void train (List<Image> images){

        for(Image img:images){
            List<double[][]> inList = new ArrayList<>();
            inList.add(multiply(img.getData(), (1.0/scaleFactor)));

            double[] out = _layers.get(0).getOutput(inList);
            double[] dldO = getErrors(out, img.getLabel());

            _layers.get((_layers.size()-1)).backPropagation(dldO);
        }

    }

    public void save(List<String> layers, List<List<Object>> params, String filename, int inRows, int inCols, double scaleFactor) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
        int size;
        try {
            for (int i = 0; i < layers.size(); i++) {
                writer.write(layers.get(i) + " ");
            }
            writer.write("\n");
            for (int i = 0; i < layers.size(); i++) {
                if (layers.get(i) == "Conv") {size = 5;} else if (layers.get(i) == "Pool") {size = 2;} else {size = 3;}
                for (int j = 0; j < size; j++) {
                    writer.write(params.get(i).get(j) + " ");
                }
                writer.write("\n");
            }
            writer.write(inRows + " " + inCols + " " + scaleFactor + "\n");
            writer.write("\n");

            if (layers.getLast() == "FC") {
                FullyConnectedLayer fc = (FullyConnectedLayer) _layers.getLast();
                double[][] weights = fc.get_svWeights();
                for (int i = 0; i < weights.length; i++) {
                    for (int j = 0; j < weights[0].length; j++) {
                        writer.write(weights[i][j] + " ");
                    }
                    writer.write("\n");
                }
                writer.write("\n");
            }

            if (layers.getFirst() == "Conv") {
                ConvolutionLayer conv = (ConvolutionLayer) _layers.getFirst();
                List<double[][]> filters = conv.get_svFilters();
                for (int k = 0; k < filters.size(); k++) {
                    double[][] filter = filters.get(k);
                    for (int i = 0; i < filter.length; i++) {
                        for (int j = 0; j < filter[0].length; j++) {
                            writer.write(filter[i][j] + " ");
                        }
                        writer.write("\n");
                    }
                    writer.write("\n");
                }
            }
        }
        finally {
            writer.close();
        }
    }

    public void load(String filename) throws FileNotFoundException {
        LinkedHashMap<String, String[]> layer_param = loadFilterNParam("ckpt");
        String[] convParams = (String[]) layer_param.get("Conv");
        String[] poolParams = (String[]) layer_param.get("Pool");
        String[] fcParams = (String[]) layer_param.get("FC");
        int skipLayers = layer_param.size() + 1;

        String[] build = loadBuilder(skipLayers, "ckpt");
        int inRows = Integer.parseInt(build[0]);
        int inCols = Integer.parseInt(build[1]);
        int skipToWeight = skipLayers + 2;

        int numFilters = Integer.parseInt(convParams[0]);
        int filterSize = Integer.parseInt(convParams[1]);
        int stepSizeC = Integer.parseInt(convParams[2]);
        int inRowsC = (inRows-filterSize)/stepSizeC +1;
        int inColsC = (inCols-filterSize)/stepSizeC +1;

        int windowSize = Integer.parseInt(poolParams[0]);
        int stepSizeP = Integer.parseInt(poolParams[1]);
        int inRowsP = (inRowsC-windowSize)/stepSizeP + 1;
        int inColsP = (inColsC-windowSize)/stepSizeP + 1;

        int outLength = Integer.parseInt(fcParams[0]);
        int inLength = numFilters*inRowsP*inColsP;
        double[][] weights = loadWeights(inLength, outLength, skipToWeight, "ckpt");
        int skipToFilters = skipToWeight + weights.length + 1;

        List<double[][]> filters = loadFilters(numFilters, filterSize, skipToFilters, "ckpt");
    }
    public double[][] loadWeights(int inLength, int outLength, int skipLines, String filename) throws FileNotFoundException{
        File file = new File(filename);
        Scanner sc = new Scanner(file);

        for (int i = 0; i < skipLines; i++) {
            sc.nextLine();
        }
        double[][] weights = new double[inLength][outLength];
        for (int i = 0; i < inLength; i++){
            String matrixWeight = sc.nextLine();
            String[] lineW = matrixWeight.split(" ");
            for (int j = 0; j < outLength; j++) {
                weights[i][j] = Double.parseDouble(lineW[j]);
            }
        }
        return weights;
    }

    public String[] loadBuilder(int skipLayers, String filename) throws FileNotFoundException {
        File file = new File(filename);
        Scanner sc = new Scanner(file);

        for (int i = 0; i < skipLayers; i++) {
            sc.nextLine();
        }
        String lineB = sc.nextLine();
        String[] build = lineB.split(" ");

        return build;
    }

    public LinkedHashMap<String, String[]> loadFilterNParam(String filename) throws FileNotFoundException {
        File file = new File(filename);
        Scanner sc = new Scanner(file);

        LinkedHashMap layer_param = new LinkedHashMap<String, String[]>();
        String line = sc.nextLine();
        String[] layers = line.split(" ");
        List<String[]> params = new ArrayList<>();
        for (int k = 0; k<layers.length; k++) {
            String lineI = sc.nextLine();
            String[] param = lineI.split(" ");
            params.add(param);
            layer_param.put(layers[k], params.get(k));
        }
        return layer_param;
    }

    public List<double[][]> loadFilters(int numFilters, int filterSize, int skipLines, String filename) throws FileNotFoundException {
        File file = new File(filename);
        Scanner sc = new Scanner(file);
        List<double[][]> filters = new ArrayList<>();

        for (int i = 0; i < skipLines; i++) {
            sc.nextLine();
        }
        for (int k = 0; k < numFilters; k++) {
            double[][] filter = new double[filterSize][filterSize];
            for (int i = 0; i < filterSize; i++) {
                String lineF = sc.nextLine();
                String[] fValue = lineF.split(" ");
                for (int j = 0; j < filterSize; j++) {
                    filter[i][j] = Double.parseDouble(fValue[j]);
                }
            }
            filters.add(filter);
            sc.nextLine();
        }
        return filters;
    }
}