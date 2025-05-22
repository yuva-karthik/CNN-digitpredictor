package src.layers;

import java.util.List;
import java.util.Random;
import java.io.Serializable;


public class FullLayer extends Layer implements Serializable{

    private long seed;
    private double[][] _weights;
    private double[] _biases;
    private int _inlength;
    private int _outlength;
    private double _learningRate;
    private double[] lastInput;
    private double[] lastZ;
    private double[][] gradWeights;
    private double[] gradBiases;
    private int batchCount = 0;

    public FullLayer(int inlen , int outlen , long seed , double _learningRate){
        this._inlength = inlen;
        this._outlength = outlen;
        this.seed = seed;
        this._learningRate = _learningRate;

        _weights = new double[inlen][outlen];
        _biases = new double[outlen];
        setWeight();
        setBias();
    }

    private double[] softmax(double[] input) {
        double max = input[0];
        for (double val : input) {
            if (val > max) max = val;
        }

        double sum = 0.0;
        double[] exp = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            exp[i] = Math.exp(input[i] - max); 
            sum += exp[i];
        }

        for (int i = 0; i < input.length; i++) {
            exp[i] /= sum;
        }

        return exp;
    }


    private double reLu(double x) {
        return (0 > x)? 0 : x; 
    }
    
    private double derivated_reLu(double x) {
        return (x > 0) ? 1 : 0.01;
    }
    
    public double[] FullForwardPass(double[] input){
        lastInput = input;    
        double[] z = new double[_outlength];
        for (int i = 0; i < _inlength; i++){
            for (int j = 0; j < _outlength; j++){
                z[j] += _weights[i][j] * input[i];
            }
        }
        for (int j = 0; j < _outlength; j++){
            z[j] += _biases[j];
        }
        lastZ = z;         
        double[] output = new double[_outlength];
        for (int j = 0; j < _outlength; j++){
            output[j] = reLu(z[j]);
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
        return (_next != null) ? _next.getOutput(softmax(result)) : softmax(result);
    }

    @Override
    public void backPropagation(double[] dldo) {
        double[] delta = new double[_outlength];
        for (int j = 0; j < _outlength; j++) {
            double deriv = (lastZ[j] > 0) ? 1 : 0.01; // Leaky ReLU
            delta[j] = dldo[j] * deriv;
        }

        for (int i = 0; i < _inlength; i++) {
            for (int j = 0; j < _outlength; j++) {
                gradWeights[i][j] += delta[j] * lastInput[i];
            }
        }
        for (int j = 0; j < _outlength; j++) {
            gradBiases[j] += delta[j];
        }

        batchCount++;

    // Backprop to previous layer
        double[] dldx = new double[_inlength];
        for (int i = 0; i < _inlength; i++) {
            for (int j = 0; j < _outlength; j++) {
                dldx[i] += _weights[i][j] * delta[j];
            }
        }
        if (_prev != null) {
            _prev.backPropagation(dldx);
        }
    }

    public void applyBatchUpdate() {
        if (batchCount == 0) return;

        double sumWBefore = 0;
        for (double[] row : _weights)
            for (double w : row)
                sumWBefore += Math.abs(w);

        for (int i = 0; i < _inlength; i++) {
            for (int j = 0; j < _outlength; j++) {
                _weights[i][j] -= (_learningRate * gradWeights[i][j]) / batchCount;
            }
        }
        for (int j = 0; j < _outlength; j++) {
            _biases[j] -= (_learningRate * gradBiases[j]) / batchCount;
        }

        double sumWAfter = 0;
        for (double[] row : _weights)
            for (double w : row)
                sumWAfter += Math.abs(w);

        System.out.printf("FullLayer batch update: |W| before=%.4f, after=%.4f, Δ=%.4f%n",
            sumWBefore, sumWAfter, sumWAfter - sumWBefore);
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

    public void resetGradients() {
        gradWeights = new double[_inlength][_outlength];
        gradBiases = new double[_outlength];
        batchCount = 0;
    }

    @Override
    public void backPropagation(List<double[][]> dldo) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagation'");
    }
    

}
