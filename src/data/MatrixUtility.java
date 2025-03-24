package src.data;

public class MatrixUtility {
    public static double[][] add(double[][] a , double[][] b){
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    public static double[] add(double a, double[] b){
        int cols = b.length;
        double[] result = new double[cols];
        for(int j = 0; j < cols; j++){
            result[j] = a + b[j];
        }
        return result;
    }

    public static double[] add(double[] a, double[] b){
        int cols = b.length;
        double[] result = new double[cols];
        for(int j = 0; j < cols; j++){
            result[j] = a[j] + b[j];
        }
        return result;
    }

    public static double[][] scalerMultiply(double[][] a , double scalar){
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                result[i][j] = a[i][j] * scalar;
            }
        }
        return result;
    }

    public static double[] scalerMultiply(double[] a , double scalar){
        int cols = a.length;
        double[] result = new double[cols];
        for(int j = 0; j < cols; j++){
            result[j] = a[j] * scalar;
        }
        return result;
    }

}
