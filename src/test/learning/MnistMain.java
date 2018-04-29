package test.learning;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.activation.Sigmoid;
import at.fhtw.ai.nn.learning.BackPropagation;
import at.fhtw.ai.nn.utils.Utils;
import test.loader.DigitImage;
import test.loader.DigitImageLoadingService;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

/**
 * Created On: 29.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class MnistMain {
    public static void main(String[] args) {
        double learningRate = 0.2;
        double momentum = 0.9;
        double meanNetworkError = 0.005;

        long startTime = System.currentTimeMillis();

        int epochCount = 0;
        int saveCount = 0;

        int numberOfInputNeurons = 28 * 28;
        int numberOfOutputNeurons = 10;
        int numberOfHiddenNeurons = (int) Math.round(Math.sqrt(numberOfInputNeurons * numberOfOutputNeurons));

        System.out.println("Setting up Neural Network...");
        Layer inputLayer = Utils.createLayer("Input Layer", numberOfInputNeurons);
        Layer outputLayer = Utils.createLayer("Output Layer", numberOfOutputNeurons);
        Layer hiddenLayer = Utils.createLayer("Hidden Layer", numberOfHiddenNeurons);

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.getLayers().add(inputLayer);
        neuralNetwork.getLayers().add(hiddenLayer);
        neuralNetwork.getLayers().add(outputLayer);
        neuralNetwork.setActivationFunctions(new Sigmoid());
        neuralNetwork.connectLayersInOrder();

        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setLearningRate(learningRate);
        backPropagation.setMomentum(momentum);
        backPropagation.setMeanSquareError(meanNetworkError);
        backPropagation.setNeuralNetwork(neuralNetwork);

        System.out.println("Loading Train Images...");
        DigitImageLoadingService dilsTrainData = new DigitImageLoadingService(
                "C:/Users/Daniel/Desktop/train/train-labels-idx1-ubyte.dat",
                "C:/Users/Daniel/Desktop/train/train-images-idx3-ubyte.dat"
        );


        List<DigitImage> trainImages = null;
        try {
            trainImages = dilsTrainData.loadDigitImages();
            Collections.shuffle(trainImages);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Pre-Initialize desired output values for back propagation
        for (int i = 0; i < neuralNetwork.getOutputLayer().getNeurons().size(); i++) {
            backPropagation.getDesiredOutputValues().add(0.0);
        }

        // Do learning here
        System.out.println("Start Learning Process...");
        int epochIterations = 0;
        double epochError = 0.0;
        double averageNetworkError;
        do {
            Collections.shuffle(trainImages);
            for (DigitImage digitImage : trainImages) {
                // Set neural network input parameters
                for (int i = 0; i < inputLayer.getNeurons().size(); i++) {
                    inputLayer.getNeurons().get(i).value = digitImage.getData()[i];
                }

                // Set desired output values
                for (int i = 0; i < backPropagation.getDesiredOutputValues().size(); i++) {
                    backPropagation.getDesiredOutputValues().set(i, 0.0);
                }
                backPropagation.getDesiredOutputValues().set(digitImage.getLabel(), 1.0);

                // Learn the network
                backPropagation.learn();

                // Update epoch counters
                epochError += backPropagation.networkError();
                epochIterations++;
            }

            // Calculate average network error
            averageNetworkError = epochError / (double) epochIterations;

            // Save the network
            System.out.println();
            System.out.println("Finished Epoch-" + epochCount);
            System.out.println("   Seconds Since Start: " + ((System.currentTimeMillis() - startTime) / 1000) + " s");
            System.out.println("   Average Network Error: " + averageNetworkError + " (" + ((int) epochError) + " / " + epochIterations + ")");
            System.out.println("   Saving Neural Network...");
            save(neuralNetwork, epochCount, saveCount);
            System.out.println("   Neural Network Successfully Saved - Pausing For 5 Seconds!");

            try {
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            // Reset epoch counters
            epochError = 0.0;
            epochIterations = 0;
            epochCount++;
            saveCount = 0;
        } while (averageNetworkError > meanNetworkError);
    }

    /**
     * Saves the neural network.
     *
     * @param neuralNetwork Neural network.
     * @param epochCount    Epoch count.
     * @param saveCount     Save count.
     */
    private static void save(NeuralNetwork neuralNetwork, int epochCount, int saveCount) {
        try {
            Utils.serialize(neuralNetwork, "C:/Users/Daniel/Desktop/pl3/NeuralNetwork_E" + epochCount + "_C" + saveCount + ".dat");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
