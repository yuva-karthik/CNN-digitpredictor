package src.layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {
    protected Layer _next;
    protected Layer _prev;

    public Layer getNextLayer(){
        return _next;
    }

    public Layer getPrevLayer(){
        return _prev;
    }

    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[] input);

    public abstract void backPropagation(List<double[][]> d);
    public abstract void backPropagation(double[] d);

    public abstract int getLength();
    public abstract int getRow();
    public abstract int getCol();
    public abstract int getElement();

    public double[] makeVector(List<double[][]> input){
        int n = input.size();
        int row = input.get(0).length;
        int col = input.get(0)[0].length;
        double[] vector = new double[n * row * col];
        int k = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < row; j++) {
                for (int l = 0; l < col; l++) {
                    vector[k++] = input.get(i)[j][l];
                }
            }
        }
        return vector;
    }

    public List<double[][]> makeMatrix(double[] input , int row , int col , int len){
        List<double[][]> matrix = new ArrayList<>();
        int k = 0;
        for (int i = 0; i < len; i++) {
            double[][] matrixRow = new double[row][col];
            for (int j = 0; j < row; j++) {
                for (int l = 0; l < col; l++) {
                    matrixRow[j][l] = input[k++];
                }
            }
            matrix.add(matrixRow);
        }
        return matrix;
    }

}
