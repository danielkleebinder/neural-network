package test;

import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.utils.Utils;
import test.loader.DigitImage;
import test.loader.DigitImageLoadingService;

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
            neuralNetwork = Utils.deserialize("C:/Users/Daniel/Desktop/NeuralNetwork.dat");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        System.out.println("Loading Test Images...");
        DigitImageLoadingService dilsTestData = new DigitImageLoadingService(
                "C:/Users/Daniel/Desktop/train/t10k-labels-idx1-ubyte.dat",
                "C:/Users/Daniel/Desktop/train/t10k-images-idx3-ubyte.dat"
        );

        List<DigitImage> testImages = null;
        try {
            testImages = dilsTestData.loadDigitImages();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Testing in Progress...");
        int correct = 0;
        for (int i = 0; i < testImages.size(); i++) {
            DigitImage digitImage = testImages.get(i);

            for (int j = 0; j < digitImage.getData().length; j++) {
                neuralNetwork.getInputLayer().getNeurons().get(j).setValue(digitImage.getData()[j]);
            }

            neuralNetwork.fireOutput();

            int highestIndex = -1;
            double highestOutput = 0.0;
            for (int k = 0; k < neuralNetwork.getOutputLayer().getNeurons().size(); k++) {
                Neuron neuron = neuralNetwork.getOutputLayer().getNeurons().get(k);
                if (highestOutput < neuron.getValue()) {
                    highestIndex = k;
                    highestOutput = neuron.getValue();
                }

            }
            if (highestIndex == digitImage.getLabel()) {
                correct++;
            }
        }
        System.out.println("Testing Done!");
        System.out.println("Total: " + testImages.size());
        System.out.println("Correct: " + correct);
        System.out.println("Wrong: " + (testImages.size() - correct));
        System.out.println("Accuracy: " + (correct / (double) testImages.size()));
    }
}
