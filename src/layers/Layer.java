package src.layers;

import java.util.ArrayList;
import java.util.List;
import java.io.Serializable;


public abstract class Layer implements Serializable{
    protected Layer _next;
    protected Layer _prev;

    public Layer getNextLayer(){
        return _next;
    }

    public Layer getPrevLayer(){
        return _prev;
    }

    public void setNextLayer(Layer next){
        this._next = next;
    }

    public void setPrevLayer(Layer prev){
        this._prev = prev;
    }

    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[] input);

    public abstract void backPropagation(List<double[][]> dldo);
    public abstract void backPropagation(double[] dldo);

    public abstract int getLength();
    public abstract int getRow();
    public abstract int getCol();
    public abstract int getElement();

    public double[] makeVector(List<double[][]> input){
        int n = input.size();
        int row = input.get(0).length;
        int col = input.get(0)[0].length;
        double[] vector = new double[n * row * col];
        int m = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < row; j++) {
                for (int k = 0; k < col; k++) {
                    vector[m++] = input.get(i)[j][k];
                }
            }
        }
        return vector;
    }

    public List<double[][]> makeMatrix(double[] input, int row, int col, int len) {
        List<double[][]> matrix = new ArrayList<>();
        int k = 0;
        for (int i = 0; i < len; i++) {
            double[][] matrixRow = new double[row][col];
            for (int j = 0; j < row; j++) {
                for (int l = 0; l < col; l++) {
                    if (k < input.length) {
                        matrixRow[j][l] = input[k++];
                    } else {
                        matrixRow[j][l] = 0; 
                    }
                }
            }
            matrix.add(matrixRow);
        }
        return matrix;
    }
    

}
