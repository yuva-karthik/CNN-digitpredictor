package src;
import src.data.*;
import src.network.NetworkBuilder;
import src.network.NeuralNetwork;

import java.util.List;
import static java.util.Collections.shuffle;

public class Main {
    public static void main(String[] args) {

        long seed = 194;
        int epoch = 10;

        System.out.println("data load intialized");

        List<Image> images_test = new Data_Reader().readData("dataset/mnist_test.csv");
        List<Image> images_train = new Data_Reader().readData("dataset/mnist_train.csv");

        System.out.println("test data size: " + images_test.size());
        System.out.println("train data size: " + images_train.size());

        NetworkBuilder builder = new NetworkBuilder(28, 28, 256*100);
        builder.addConvolutionalLayer(8, 5, 1, 0.1, 194);
        builder.addMaxpoolLayer(3, 2);
        builder.addFullLayer(10, 0.1, seed);

        NeuralNetwork net = builder.build();

        float rate = net.test(images_test);
        System.out.println("training success rate: " + rate);

        for(int i = 1 ; i <= epoch ; i++){
            shuffle(images_train);
            net.train(images_train);
            rate = net.test(images_train);
            System.out.println("epoch: " + i + " success rate: " + rate);
        }
    }
    
}
