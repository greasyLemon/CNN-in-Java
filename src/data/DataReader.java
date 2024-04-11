package data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DataReader {

    private final int rows = 28;
    private final int cols = 28;

    public List<Image> readData(String path){

        List<Image> images = new ArrayList<>();

        try (BufferedReader dataReader = new BufferedReader(new FileReader(path))){

            String line;

            while((line = dataReader.readLine()) != null){
                String[] lineItems = line.split(",");

                double[][] data = new double[rows][cols];
                int label = Integer.parseInt(lineItems[0]);

                int i = 1;

                for(int row = 0; row < rows; row++){
                    for(int col = 0; col < cols; col++){
                        data[row][col] = (double) Integer.parseInt(lineItems[i]);
                        i++;
                    }
                }

                images.add(new Image(data, label));

            }

        } catch (Exception e){
            throw new IllegalArgumentException("File not found " + path);
        }

        return images;

    }

    public static double[][] loadImage(String path) throws FileNotFoundException {
        File file = new File(path);
        Scanner sc = new Scanner(file);
        double[][] img = new double[28][28];
        for (int i = 0; i< 28; i++) {
            String line = sc.nextLine();
            String[] value = line.split(" ");
            for (int j =0; j <28; j++) {
                img[i][j] = Double.parseDouble(value[j]);
            }
        }
        return img;
    }


}