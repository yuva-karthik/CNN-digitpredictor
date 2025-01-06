package src;
import src.data.*;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        List<Image> images = new Data_Reader().readData("dataset/mnist_test");
    }
    
}
