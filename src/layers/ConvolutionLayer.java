package src.layers;
import src.data.MatrixUtility;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.io.Serializable;


public class ConvolutionLayer extends Layer implements Serializable{

    private List<double[][]> _filters = new ArrayList<double[][]>();
    private long seed;
    private int _filterSize;
    private int _step;
    private int _inlength;
    private int _inrows;
    private int _incols;
    private double _learningRate;
    

    List <double[][]> _lastinput;

    public ConvolutionLayer(int _filterSize , int _step , long seed , int _inlength , int _inrows , int _incols , int numfilter , double learningRate){
        this._filterSize = _filterSize;
        this.seed = seed;
        this._step = _step;
        this._inlength = _inlength;
        this._inrows = _inrows;
        this._incols = _incols;
        this._learningRate = learningRate;
        makeFilter(numfilter);
    }

    List<double[][]> ConvolutionalForwardPass(List<double[][]> list){
        _lastinput = list;
        List<double[][]> output = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            for(double[][] f : _filters)
            output.add(convolve(list.get(i) , f , _step));
        }
        return output;
    }

    private double[][] convolve(double[][] input , double[][] filter , int step){
        int inrow = input.length;
        int incol = input[0].length;
    
        int frow = filter.length;
        int fcol = filter[0].length;
    
        int outrow = (inrow - frow)/step + 1;
        int outcol = (incol - fcol)/step + 1;
    
        double[][] output = new double[outrow][outcol];
        int r = 0 , c = 0;
    
        for(int i = 0 ; i <= inrow - frow ; i+=step){
            c = 0;
            for(int j = 0 ; j <= incol - fcol ; j+=step){
                double sum = 0;
                for(int k = 0 ; k < frow ; k++){
                    for(int l = 0 ; l < fcol ; l++){
                        sum += filter[k][l] * input[i+k][j+l];
                    }
                }
                output[r][c] = sum;
                c++;
            }
            r++;
        }
        return output;
    }
    
    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = ConvolutionalForwardPass(input);
        return _next.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> output = makeMatrix(input, _inrows, _incols, _inlength);
        return getOutput(output);
    }

    double[][] spaceArray(double[][] input){
        if(_step == 1){
            return input;
        }
        int inrow = (input.length - 1) * _step + 1;
        int incol = (input[0].length - 1) * _step + 1;

        double[][] output = new double[inrow][incol];
        for(int i = 0 ; i < input.length ; i++){
            for(int j = 0 ; j < input[0].length ; j++){
                output[i*_step][j*_step] = input[i][j];
            }
        }
        return output;
    }
    @Override
    public void backPropagation(List<double[][]> dldo) {
        List<double[][]> fdelta = new ArrayList<>();
        List<double[][]> prevdldo = new ArrayList<>();

        for(int f = 0 ; f < _filters.size() ; f++){
            fdelta.add(new double[_filterSize][_filterSize]);
        }

        for (int i = 0; i < _lastinput.size(); i++) {
            double[][] errorinput = new double[_inrows][_incols];
            for (int j = 0; j < _filters.size(); j++) {
                double[][] error = dldo.get(i * _filters.size() + j);
                double[][] spaceError = spaceArray(error);
        
                double[][] dldf = convolve(spaceError, _filters.get(j), 1);
                double[][] delta = MatrixUtility.scalerMultiply(dldf, -_learningRate);
                fdelta.set(j, MatrixUtility.add(fdelta.get(j), delta));
        
                double[][] rotated = flipHorizontal(flipVertical(_filters.get(j)));
                double[][] backGrad = completeConvolve(spaceError, rotated);
                errorinput = MatrixUtility.add(errorinput, backGrad);
            }
            prevdldo.add(errorinput);
        }
        for(int j = 0 ; j < _filters.size() ; j++){
            double[][] modified = MatrixUtility.add(fdelta.get(j),_filters.get(j));
            _filters.set(j,modified);
        }
        if(_prev != null){
            _prev.backPropagation(prevdldo);
        }
    }

    @Override
    public void backPropagation(double[] dldo) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagation'");
    }

    private double[][] completeConvolve(double[][] input, double[][] filter) {
        int inRows = input.length;
        int inCols = input[0].length;
        int fRows  = filter.length;
        int fCols  = filter[0].length;
    
        int outRows = inRows + fRows - 1;
        int outCols = inCols + fCols - 1;
        double[][] output = new double[outRows][outCols];
    

        for (int i = 0; i < outRows; i++) {
            for (int j = 0; j < outCols; j++) {
                double sum = 0;
 
                for (int k = 0; k < fRows; k++) {
                    for (int l = 0; l < fCols; l++) {
                        int inI = i - k;
                        int inJ = j - l;

                        if (inI >= 0 && inI < inRows && inJ >= 0 && inJ < inCols) {
                            sum += filter[k][l] * input[inI][inJ];
                        }
                    }
                }
                output[i][j] = sum;
            }
        }
        return output;
    }

    @Override
    public int getLength() {
        return _filters.size()*_inlength;
    }

    @Override
    public int getRow() {
        return (_inrows - _filterSize)/_step + 1;
    }

    @Override
    public int getCol() {
        return (_incols - _filterSize)/_step + 1;
    }

    @Override
    public int getElement() {
        return getLength()*getRow()*getCol();
    }

    private void makeFilter(int n){
        List<double[][]> filterset = new ArrayList<>();
        Random rn = new Random(seed);
        for (int i = 0; i < n; i++) {
            double[][] filter = new double[_filterSize][_filterSize];
            for (int j = 0; j < _filterSize; j++) {
                for (int k = 0; k < _filterSize; k++) {
                    filter[j][k] = rn.nextGaussian();  
                }
            }
            filterset.add(filter);
        }
        _filters = filterset;
    }

    public double[][] flipHorizontal(double[][] a){
        int rows = a.length;
        int cols = a[0].length;
        double[][] b = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                b[i][cols - 1 - j] = a[i][j];
            }
        }
        return b;
    }

    public double[][] flipVertical(double[][] a){
        int rows = a.length;
        int cols = a[0].length;
        double[][] b = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                b[rows - 1 - i][j] = a[i][j];
            }
        }
        return b;
    }
}
