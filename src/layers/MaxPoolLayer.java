package src.layers;

import java.util.ArrayList;
import java.util.List;
import java.io.Serializable;


public class MaxPoolLayer extends Layer implements Serializable{

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

    public List<double[][]> MaxForwardPass(List<double[][]> input) {
        _lastmaxrow.clear();
        _lastmaxcol.clear();
        List<double[][]> output = new ArrayList<>();
        for (double[][] in : input) {
            output.add(pool(in));
        }
        return output;
    }

    private double[][] pool(double[][] input) {
        int outRow = getRow();
        int outCol = getCol();
        double[][] output = new double[outRow][outCol];
    
        int[][] maxRow = new int[outRow][outCol];
        int[][] maxCol = new int[outRow][outCol];
    
        for (int i = 0; i < outRow; i++) {
            for (int j = 0; j < outCol; j++) {
                double max = Double.NEGATIVE_INFINITY;
                maxRow[i][j] = -1;
                maxCol[i][j] = -1;
                for (int m = 0; m < _window; m++) {
                    for (int n = 0; n < _window; n++) {
                        int inR = i * _step + m;
                        int inC = j * _step + n;
                        if (input[inR][inC] > max) {
                            max = input[inR][inC];
                            maxRow[i][j] = inR;
                            maxCol[i][j] = inC;
                        }
                    }
                }
                output[i][j] = max;
            }
        }
        _lastmaxrow.add(maxRow);
        _lastmaxcol.add(maxCol);
        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outpool = MaxForwardPass(input);
        return _next.getOutput(outpool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matinput = makeMatrix(input, _inrows, _incols, _inlength);
        return getOutput(matinput);
    }

    @Override
    public void backPropagation(List<double[][]> dldo) {
        List<double[][]> dxdl = new ArrayList<>();
        int k = 0;
        for(double[][] arr : dldo){
            double[][] error = new double[_inrows][_incols];
            for(int i = 0 ; i < getRow() ; i++){
                for(int j = 0 ; j < getCol() ; j++){
                    int max_i = _lastmaxrow.get(k)[i][j];
                    int max_j = _lastmaxcol.get(k)[i][j];

                    if(max_i != -1){
                        error[max_i][max_j] += arr[i][j];
                    }
                }
            }
            dxdl.add(error);
            k++;
        }
        if(_prev != null){
            _prev.backPropagation(dxdl);
        }
    }

    @Override
    public void backPropagation(double[] dldo) {
        int expectedSize = getLength() * getRow() * getCol();
        if (dldo.length != expectedSize) {
            System.err.println("MaxPoolLayer gradient size mismatch: expected "
                + expectedSize + " but got " + dldo.length);
            throw new IllegalArgumentException("MaxPoolLayer backPropagation(): dimension mismatch");
        }
        List<double[][]> matrix = makeMatrix(dldo, getRow(), getCol(), getLength());
        backPropagation(matrix);
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
