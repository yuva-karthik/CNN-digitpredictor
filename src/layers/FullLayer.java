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
        double[] vector = makeVector(dldo);
        backPropagation(vector);
    }

    @Override
    public void backPropagation(double[] dldo) {
        double[] dldx = new double[_inlength];
        double[] dodx = new double[_outlength];
        double[][] dxdw = new double[_inlength][_outlength];
        
        for(int i = 0 ; i < _outlength ; i++){
            dodx[i] = derivated_reLu(dldo[i]);
        }

        for(int i = 0 ; i < _inlength ; i++){
            for(int j = 0 ; j < _outlength ; j++){
                dldx[i] += dldo[j] * dodx[j] * _weights[i][j];
            }
        }

        for(int i = 0 ; i < _inlength ; i++){
            for(int j = 0 ; j < _outlength ; j++){
                dxdw[i][j] = dldo[j] * dodx[j];
            }
        }

        for(int i = 0 ; i < _inlength ; i++){
            for(int j = 0 ; j < _outlength ; j++){
                _weights[i][j] -= dxdw[i][j];
            }
        }

        for(int i = 0 ; i < _outlength ; i++){
            _biases[i] -= dodx[i];
        }

        if(_prev != null){
            _prev.backPropagation(dldx);
        }
    }

    @Override
    public int getLength() {
        return 0;
    }

    @Override
    public int getRow() {
        return 0;
    }

    @Override
    public int getCol() {
        return 0;
    }

    @Override
    public int getElement() {
        return _outlength;
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
