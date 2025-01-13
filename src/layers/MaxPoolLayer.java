package src.layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer {

    private int _step;
    private int _window;
    private int _inlength;
    private int _inrows;
    private int _incols;

    private List<int[][]> _lastmaxrow = new ArrayList<>();
    private List<int[][]> _lastmaxcol = new ArrayList<>();

    public MaxPoolLayer(int step , int window , int length , int rows , int cols){
        this._step = step;
        this._window = window;
        this._inlength = length;
        this._inrows = rows;
        this._incols = cols;
    }

    public List<double[][]> MaxForwardPass(List<double[][]> input){
        List<double[][]> output = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {
            output.add(pool(input.get(i)));
        }
        return output;
    }

    private double[][] pool(double[][] input){
        int outrow = getRow();
        int outcol = getCol();

        double[][] output = new double[outrow][outcol];
        int[][] maxrow = new int[outrow][outcol];
        int[][] maxcol = new int[outrow][outcol];

        for (int i = 0; i < outrow; i+=_step) {
            for (int j = 0; j < outcol; j+=_step) {
                double max = 0.0;
                maxrow[i][j] = -1;
                maxcol[i][j] = -1;
                for (int k = 0; k < _window; k++) {
                    for (int l = 0; l < _window; l++) {
                        if(input[i+k][j+l] > max){
                            max = input[i+k][j+l];
                            maxrow[i][j] = i+k;
                            maxcol[i][j] = j+l;
                        }
                    }
                }
                output[i][j] = max;
            }
        }
        _lastmaxrow.add(maxrow);
        _lastmaxcol.add(maxcol);
        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutput'");
    }

    @Override
    public double[] getOutput(double[] input) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutput'");
    }

    @Override
    public void backPropagation(List<double[][]> dldo) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagation'");
    }

    @Override
    public void backPropagation(double[] dldo) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagation'");
    }

    @Override
    public int getLength() {
        return _inlength;
    }

    @Override
    public int getRow() {
        return (_inrows - _window) / _step + 1;
    }

    @Override
    public int getCol() {
        return (_incols - _window) / _step + 1;
    }

    @Override
    public int getElement() {
        return _inlength*getCol()*getRow();
    }
    
}
