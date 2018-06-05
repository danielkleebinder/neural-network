package test;

import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.utils.Utils;
import test.loader.GrayImage;
import test.loader.MNISTImageLoadingService;
import test.loader.MnistImage;

import java.io.IOException;
import java.util.List;

/**
 * TrainMain program entry point.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class TestMain {
    public static void main(String[] args) {
        System.out.println("Loading Neural Network...");
        NeuralNetwork neuralNetwork = null;
        try {
            //neuralNetwork = Utils.deserialize("../NeuralNetwork_E20_97_74.dat");
            neuralNetwork = Utils.deserialize("C:/Users/Daniel/Desktop/pl3/NeuralNetwork_E10_C0.dat");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        System.out.println("Loading Test Images...");
        MNISTImageLoadingService dilsTestData = new MNISTImageLoadingService(
                "C:/Users/Daniel/Desktop/train/t10k-labels-idx1-ubyte.dat",
                "C:/Users/Daniel/Desktop/train/t10k-images-idx3-ubyte.dat"
        );

        List<MnistImage> testImages = null;
        try {
            testImages = dilsTestData.loadMNISTImages(new GrayImage.Factory());
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Testing in Progress...");
        int[][] confusionMatrix = new int[10][10];
        int correct = 0;
        for (int i = 0; i < testImages.size(); i++) {
            MnistImage mnistImage = testImages.get(i);
            neuralNetwork.input(mnistImage.getData());
            neuralNetwork.fireOutput();

            int highestIndex = -1;
            double highestOutput = -1.0;
            double[] outputs = neuralNetwork.output();
            for (int k = 0; k < outputs.length; k++) {
                if (highestOutput < outputs[k]) {
                    highestIndex = k;
                    highestOutput = outputs[k];
                }
            }
            if (highestIndex == mnistImage.getLabel()) {
                correct++;
            }
            confusionMatrix[highestIndex][mnistImage.getLabel()]++;
        }

        double accuracy = correct / (double) testImages.size();
        double errorRate = 1.0 - accuracy;

        System.out.println("Testing Done!");
        printConfusionMatrix(confusionMatrix);
        System.out.println("Total Images Tested: " + testImages.size());
        System.out.println("Accuracy: " + accuracy + " (Correct: " + correct + ")");
        System.out.println("Error Rate: " + errorRate + " (Wrong: " + (testImages.size() - correct) + ")");
    }

    private static void printConfusionMatrix(int[][] confusionMatrix) {
        StringBuilder result = new StringBuilder();
        result.append("      ");
        for (int i = 0; i < confusionMatrix.length; i++) {
            result.append(String.format("%-8d", i));
        }
        result.append('\n');
        for (int i = 0; i < confusionMatrix.length; i++) {
            result.append("--------");
        }
        result.append('\n');
        for (int x = 0; x < confusionMatrix.length; x++) {
            result.append(x).append("  │  ");
            for (int y = 0; y < confusionMatrix[x].length; y++) {
                result.append(String.format("%-8d", confusionMatrix[x][y]));
            }
            result.append('\n');
        }
        result.append("F  │  ");
        for (int i = 0; i < confusionMatrix.length; i++) {
            int sum = 0;
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                if (i == j) {
                    continue;
                }
                sum += confusionMatrix[j][i];
            }
            result.append(String.format("%-8d", sum));
        }
        result.append('\n');
        System.out.println(result.toString());
    }
}