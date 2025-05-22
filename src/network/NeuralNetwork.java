package src.network;
import src.layers.FullLayer;
import src.layers.Layer;
import src.data.Image;
import src.data.MatrixUtility;

import java.io.*;
import java.util.List;
import java.util.ArrayList;

public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;
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

        double[] error = new double[netOutput.length];
        for (int i = 0; i < error.length; i++) {
            error[i] = netOutput[i] - expected[i];
        }
        return error;
    }

    private int getMaxIndex(double[] in){
        double max = in[0];
        int index = 0;
        for(int i = 0 ; i < in.length ; i++){
            if(in[i] > max){
                max = in[i];
                index = i;
            }
        }
        return index;
    }

    int guess(Image image) {
        List<double[][]> inlist = new ArrayList<>();
        inlist.add(MatrixUtility.scalerMultiply(image.getData(), 1.0/scalefactor));
        double[] logits = _layers.get(0).getOutput(inlist);
        double[] probs = MatrixUtility.softmax(logits);
        return getMaxIndex(probs);
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

    public double trainOnImage(Image img) {
        List<double[][]> inlist = new ArrayList<>();
        inlist.add(
            MatrixUtility.scalerMultiply(img.getData(), 1.0 / scalefactor)
        );
        double[] logits = _layers.get(0).getOutput(inlist);
        double[] probs  = MatrixUtility.softmax(logits);
        double pTrue = probs[img.getLabel()];
        double loss = -Math.log(pTrue + 1e-12);
        double[] dldo = getErrors(probs, img.getLabel());
        _layers.get(_layers.size() - 1).backPropagation(dldo);
    
        return loss;
    }

    public double train(List<Image> images, int batchSize) {
    double totalLoss = 0;
    int count = 0;
    for (Layer layer : _layers) {
        if (layer instanceof FullLayer) {
            ((FullLayer) layer).resetGradients();
        }
    }
    for (Image img : images) {
        totalLoss += trainOnImage(img);
        count++;
        if (count % batchSize == 0) {
            for (Layer layer : _layers) {
                if (layer instanceof FullLayer) {
                    ((FullLayer) layer).applyBatchUpdate();
                    ((FullLayer) layer).resetGradients();
                }
            }
        }
    }
    if (count % batchSize != 0) {
        for (Layer layer : _layers) {
            if (layer instanceof FullLayer) {
                ((FullLayer) layer).applyBatchUpdate();
                ((FullLayer) layer).resetGradients();
            }
        }
    }

    return totalLoss / images.size();
}


    public void saveModel(String filePath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
            System.out.println("Model saved successfully.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static NeuralNetwork loadModel(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (NeuralNetwork) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }
}