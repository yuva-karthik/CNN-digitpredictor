package src.network;
import src.layers.*;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
    private NeuralNetwork net;
    private int _inputrows;
    private int _inputcols;
    private int _scalefactor;
    List<Layer> _layers = new ArrayList<>();

    public NetworkBuilder(int rows , int cols , int scalefactor){
        this._inputrows = rows;
        this._inputcols = cols;
        this._scalefactor = scalefactor;
    }

    public void addConvolutionalLayer(int numfilter , int filterSize , int step , double learningRate , long seed){
        if(_layers.isEmpty()){
            _layers.add(new ConvolutionLayer(filterSize, step, seed, 1, _inputrows, _inputcols, numfilter, learningRate));
        }else{
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new ConvolutionLayer(filterSize, step, seed, prev.getLength(), prev.getRow(), prev.getCol(), numfilter, learningRate));
        }
    }

    public void addMaxpoolLayer(int winsize , int step){
        if(_layers.isEmpty()){
            _layers.add(new MaxPoolLayer(step,winsize,1,_inputrows,_inputcols));
        }else{
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new MaxPoolLayer(step,winsize,prev.getLength(),prev.getRow(),prev.getCol()));
        }
    }

    public void addFullLayer(int outlength , double learningRate , long seed){
        if(_layers.isEmpty()){
            _layers.add(new FullLayer(1,outlength,seed,learningRate));
        }else{
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new FullLayer(prev.getElement(),outlength,seed,learningRate));
        }
    }

    public NeuralNetwork build(){
        net = new NeuralNetwork(_layers, _scalefactor);
        return net;
    }

}
