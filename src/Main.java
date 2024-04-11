import data.DataReader;
import data.Image;
import network.NetworkBuilder;
import network.NeuralNetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Scanner;

import static java.util.Collections.shuffle;

public class Main {

    public static List<List<Image>> subBatch(List<Image> imageSet, int batchSize) {
        List<List<Image>> batch = new ArrayList<>();
        for (int i = 0; i < imageSet.size()/batchSize; i++) {
            batch.add(imageSet.subList(i*batchSize, (i+1)*batchSize));
        }
        batch.add(imageSet.subList((imageSet.size()/batchSize)*batchSize, imageSet.size()));
        return batch;
    }

    public static void main(String[] args) throws IOException {
        int batchSize = 6000;
        long SEED = 123;

        System.out.println("Starting data loading...");

        List<Image> imagesTest = new DataReader().readData("data/mnist_test.csv");
        List<Image> imagesTrain = new DataReader().readData("data/mnist_train.csv");
        //System.out.println(imagesTest.get(47));

        System.out.println("Images Train size: " + imagesTrain.size());
        System.out.println("Images Test size: " + imagesTest.size());

//        NetworkBuilder builder = new NetworkBuilder(28,28,256*100);
//        builder.addConvolutionLayer(8, 5, 1, 0.1, SEED);
//        builder.addMaxPoolLayer(3,2);
//        builder.addFullyConnectedLayer(10, 0.1, SEED);
//
//        NeuralNetwork net = builder.build();
//
//        float rate = net.test(imagesTest);
//        System.out.println("Pre training success rate: " + rate);
//
//        int epochs = 5;
//
//        shuffle(imagesTrain);
//        List<List<Image>> batch = subBatch(imagesTrain, batchSize);
//        for(int i = 0; i < epochs; i++){
//            if (i%(imagesTrain.size()/batchSize)==0 && i>=(imagesTrain.size()/batchSize)) {
//                shuffle(imagesTrain);
//                batch.clear();
//                batch = subBatch(imagesTrain, batchSize);
//            }
//            List<Image> batchIndex = batch.get(i%(imagesTrain.size()/batchSize));
//            net.train(batchIndex);
//            rate = net.test(imagesTest);
//            System.out.println("Success rate after round " + (i+1) + ": " + rate);
//        }
//
//        List<List<Object>> params = builder.svParams;
//        List<String> layers = builder.svLayers;
//        net.save(layers, params, "ckpt", 28, 28, 256*100);

        NeuralNetwork network = new NeuralNetwork();
        NeuralNetwork net = network.load(3,"ckpt");
        int result = net.guess(imagesTest.get(4567));
        System.out.println("Number: " + imagesTest.get(4567).getLabel());
        System.out.println("Predict: " + result);

    }
}