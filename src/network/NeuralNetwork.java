package src.network;
import src.layers.Layer;
import src.data.Image;
import src.data.MatrixUtility;

import java.util.List;
import java.util.ArrayList;

public class NeuralNetwork {
    List<Layer> _layers;
    double scalefactor;

    public NeuralNetwork(List<Layer> _layers , double scalefactor) {
        this._layers = _layers;
        this.scalefactor = scalefactor;
        linkLayer();
    }

    private void linkLayer(){
        if(_layers.size() <= 1){
            return;
        }
        for(int i = 0; i < _layers.size() ; i++){
            if(i == 0){
                _layers.get(i).setNextLayer(_layers.get(i+1));
            }else if(i == _layers.size()-1){
                _layers.get(i).setPrevLayer(_layers.get(i-1));
            }else{
                _layers.get(i).setNextLayer(_layers.get(i+1));
                _layers.get(i).setPrevLayer(_layers.get(i-1));
            }
        }
    }

    public double[] getErrors(double[] netOutput , int answer){
        int numclasses = netOutput.length;
        double[] expected = new double[numclasses];
        expected[answer] = 1;
        return MatrixUtility.add(netOutput , MatrixUtility.scalerMultiply(expected, -1));
    }

    private int getMaxIndex(double[] in){
        double max = 0;
        int index = 0;
        for(int i = 0 ; i < in.length ; i++){
            if(in[i] > max){
                max = in[i];
                index = i;
            }
        }
        return index;
    }

    int guess(Image image){
        List<double[][]> inlist = new ArrayList<>();
        inlist.add(MatrixUtility.scalerMultiply(image.getData(), 1.0/scalefactor));
        double[] out = _layers.get(0).getOutput(inlist);
        return getMaxIndex(out);
    }

    public float test(List<Image> images){
        int correct = 0;
        for(Image image : images){
            if(guess(image) == image.getLabel()){
                correct++;
            }
        }
        return ((float)correct/images.size());
    }

    public void train(List<Image> images){
        for(Image image : images){
            List<double[][]> inlist = new ArrayList<>();
            inlist.add(MatrixUtility.scalerMultiply(image.getData(), 1.0/scalefactor));
            double[] out = _layers.get(0).getOutput(inlist);
            double[] dldo = getErrors(out , image.getLabel());

            _layers.get(_layers.size()-1).backPropagation(dldo);
        }
    }
}
