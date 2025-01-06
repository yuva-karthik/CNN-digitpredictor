package src.data;

public class Image{
    private double[][] data;
    private int label;

    public Image(double[][] d , int l){
        this.data = d;
        this.label = l;
    }

    public double[][] getData(){
        return this.data;
    }

    public int getLabel(){
        return this.label;
    }

}