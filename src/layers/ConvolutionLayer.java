package src.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionLayer extends Layer{

    private List<double[][]> _filters = new ArrayList<double[][]>();
    private long seed;
    private int _filterSize;
    private int _step;
    private int _inlength;
    private int _inrows;
    private int _incols;

    public ConvolutionLayer(int _filterSize , int _step , int seed , int _inlength , int _inrows , int _incols , int numfilter){
        this._filterSize = _filterSize;
        this.seed = seed;
        this._step = _step;
        this._inlength = _inlength;
        this._inrows = _inrows;
        this._incols = _incols;
        makeFilter(numfilter);
    }

    List<double[][]> ConvolutionalForwardPass(List<double[][]> list){
        List<double[][]> output = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            for(double[][] f : _filters)
            output.add(convolve(list.get(i) , f));
        }
        return output;
    }

    private double[][] convolve(double[][] input , double[][] filter){
        int inrow = input.length;
        int incol = input[0].length;

        int frow = filter.length;
        int fcol = filter[0].length;

        int outrow = (inrow - frow)/_step + 1;
        int outcol = (incol - fcol)/_step + 1;

        double[][] output = new double[outrow][outcol];
        int r = 0 , c = 0;

        for(int i = 0 ; i < inrow - frow ; i+=_step){
            c = 0;
            for(int j = 0 ; j < incol - fcol ; j+=_step){
                double sum = 0;
                for(int k = 0 ; k < frow ; k++){
                    for(int l = 0 ; l < fcol ; l++){
                        sum += filter[i][j] * input[i+k][j+l];
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
                    filter[i][j] = rn.nextGaussian();
                }
            }
            filterset.add(filter);
        }
        _filters = filterset;
    }

}
