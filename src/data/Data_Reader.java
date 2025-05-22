package src.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class Data_Reader {
    private final int row = 28;
    private final int col = 28;

    public List<Image> readData(String path){
        List<Image> images = new ArrayList<>();
        try(BufferedReader d_read = new BufferedReader(new FileReader((path)))){
            String line;
            while((line = d_read.readLine()) != null){
                String[] item = line.split(",");
                double[][] data = new double[row][col];
                int label = Integer.parseInt(item[0]);

                int k = 1;
                for(int i = 0 ; i < row ; i++){
                    for(int j = 0 ; j < col ; j++){
                        data[i][j] = Double.parseDouble(item[k++]);
                    }
                }
                
                images.add(new Image(data,label));
            } 
        }catch(Exception e){
            e.printStackTrace();
        }
        return images;
    }
    
}
