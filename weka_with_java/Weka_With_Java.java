/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package weka_with_java;

/**
 *
 * @author ASUS
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStream;
import weka.core.Instances;
import weka.associations.Apriori;
import weka.associations.FPGrowth;

public class Weka_With_Java {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        Instances data = new Instances(new BufferedReader(new FileReader("dataset/supermarket.arff")));
        
//        Apriori model =  new Apriori();     // for Apriori Algorithmn

        FPGrowth model = new FPGrowth();  //FPGrowth Algorithmn
         
        model.buildAssociations(data);
        System.out.println(model);
    }
    
}
