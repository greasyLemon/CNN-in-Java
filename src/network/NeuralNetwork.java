package network;

import data.Image;
import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

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

    public void save(List<String> layers, List<List<Object>> params, String filename) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
        for (int i = 0; i < layers.size(); i++) {
            System.out.println(layers.get(i));
            System.out.println(params.get(i));
        }
        try {
            if (layers.getLast() == "FC") {
                FullyConnectedLayer fc = (FullyConnectedLayer) _layers.getLast();
                double[][] weights = fc.get_svWeights();
                for (int i = 0; i < weights.length; i++) {
                    for (int j = 0; j < weights[0].length; j++) {
                        System.out.print(weights[i][j] + " ");
                        writer.write(weights[i][j] + " ");
                    }
                    System.out.println();
                    writer.write("\n");
                }
            }
        }
        finally {
            writer.close();
        }
    }

}