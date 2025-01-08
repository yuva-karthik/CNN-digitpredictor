package src.layers;

import java.util.List;
import java.util.Random;

public class FullLayer extends Layer {

    private long seed;
    private double[][] _weights;
    private double[] _biases;
    private int _inlength;
    private int _outlength;

    public FullLayer(int inlen , int outlen , int seed){
        this._inlength = inlen;
        this._outlength = outlen;
        this.seed = seed;

        _weights = new double[inlen][outlen];
        _biases = new double[outlen];
        setWeight();
        setBias();
    }

    private double reLu(double x) {
        return (0 > x)? 0 : x; 
    }
    
    private double derivated_reLu(double x) {
        return (x > 0) ? 1 : 0.01;
    }
    
    public double[] FullForwardPass(double[] input){
        double[] output = new double[_outlength];
        for (int i = 0; i < _inlength; i++) {
            for (int j = 0; j < _outlength; j++) {
                output[j] += _weights[i][j] * input[i];
            }
        }

        for (int j = 0; j < _outlength; j++) {
            output[j] += _biases[j]; 
        }

        for (int i = 0; i < _outlength; i++) {
            output[i] = reLu(output[i]);
        }
        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = makeVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] result = FullForwardPass(input);
        if(_next != null){
            return _next.getOutput(result);
        }
        return result;
    }

    @Override
    public void backPropagation(List<double[][]> dldo) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getLength'");
    }

    @Override
    public void backPropagation(double[] dldo) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getLength'");
    }

    @Override
    public int getLength() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getLength'");
    }

    @Override
    public int getRow() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getRow'");
    }

    @Override
    public int getCol() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getCol'");
    }

    @Override
    public int getElement() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getElement'");
    }
    
    private void setWeight(){
        Random rn = new Random(seed);
        for(int i = 0; i < _inlength; i++){
            for(int j = 0; j < _outlength; j++){
                _weights[i][j] = rn.nextGaussian();
            }
        }
    }

    private void setBias(){
        Random rn = new Random(seed);
        for(int i = 0; i < _outlength; i++){
            _biases[i] = rn.nextGaussian();
        }
    }

}
