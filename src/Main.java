import data.DataReader;
import data.Image;
import network.NetworkBuilder;
import network.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

import static java.util.Collections.shuffle;

public class Main {

    public static void main(String[] args) {

        long SEED = 123;
        int batchSize = 6000;

        System.out.println("Starting data loading...");

        List<Image> imagesTest = new DataReader().readData("data/mnist_test.csv");
        List<Image> imagesTrain = new DataReader().readData("data/mnist_train.csv");
        shuffle(imagesTrain);
        List<List<Image>> batch = new ArrayList<>();
        for (int i = 0; i < imagesTrain.size()/batchSize; i++) {
            batch.add(imagesTrain.subList(i*batchSize, (i+1)*batchSize));
        }

        System.out.println("Images Train size: " + imagesTrain.size());
        System.out.println("Images Test size: " + imagesTest.size());

        NetworkBuilder builder = new NetworkBuilder(28, 28, 256*100);
        builder.addConvolutionLayer(SEED, 5, 1, 0.1, 8);
        builder.addMaxPoolLayer(2, 3);
        builder.addFullyConnectedLayer(10, 0.1, SEED);

        NeuralNetwork net = builder.build();

        float rate = net.test(imagesTest);
        System.out.println("Pre training success rate: " + rate);

        int epochs = 3;

        for(int i = 0; i < epochs; i++){
            List<Image> batchIndex = batch.get(i);
            net.train(batchIndex);
            rate = net.test(batchIndex);
            System.out.println("Success rate after round " + i + ": " + rate);
        }
    }
}