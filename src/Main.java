package src;
import src.data.*;
import src.network.NetworkBuilder;
import src.network.NeuralNetwork;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;


import java.util.List;
import java.util.Collections;

public class Main {
    public static void main(String[] args) {

        long seed = 194;
        int batchSize = 16; 

        System.out.println("data load initialized");

        List<Image> images_test = new Data_Reader().readData("F:\\CNN\\dataset\\mnist_test.csv");
        List<Image> images_train = new Data_Reader().readData("F:\\CNN\\dataset\\mnist_train.csv");

        System.out.println("test data size: " + images_test.size());
        System.out.println("train data size: " + images_train.size());

        NetworkBuilder builder = new NetworkBuilder(28, 28, 256 * 100);
        builder.addConvolutionalLayer(8, 5, 1, 0.001, seed);
        builder.addMaxpoolLayer(3, 2);
        builder.addFullLayer(10, 0.001, seed);

        NeuralNetwork net = builder.build();

        float rate = net.test(images_test);
        System.out.println("Initial test accuracy: " + rate);

        int maxEpochs = 50;
        double minDelta = 0.0001;  // minimum change to consider it an improvement
        int patience = 5;          // how many stagnant epochs to wait before stopping
        int wait = 0;
        double prevLoss = Double.MAX_VALUE;

        try (BufferedWriter writer = new BufferedWriter(new FileWriter("training_log.txt"))) {
            for (int i = 1; i <= maxEpochs; i++) {
                Collections.shuffle(images_train);
                double avgLoss = net.train(images_train, batchSize);
                float trainAcc = net.test(images_train);
                float testAcc = net.test(images_test);

                String log = String.format("Epoch %d — avg loss: %.4f | train acc: %.4f | test acc: %.4f",
                        i, avgLoss, trainAcc, testAcc);

                System.out.println(log);
                writer.write(log);
                writer.newLine();
                writer.flush();

                if (prevLoss - avgLoss < minDelta) {
                    wait++;
                    System.out.printf("No significant loss improvement (%.6f). Wait %d/%d%n", prevLoss - avgLoss, wait, patience);
                    if (wait >= patience) {
                        System.out.println("Convergence reached. Stopping training early.");
                        break;
                    }
                } else {
                    wait = 0;
                }

                prevLoss = avgLoss;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        String modelPath = "cnn_model.ser";
        net.saveModel(modelPath);
        System.out.println("Model saved to " + modelPath);
    }
}
