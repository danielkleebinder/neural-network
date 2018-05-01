package test.learning;

import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.activation.Sigmoid;
import at.fhtw.ai.nn.initialize.XavierInitializer;
import at.fhtw.ai.nn.learning.BackPropagation;
import at.fhtw.ai.nn.utils.BackPropagationBuilder;
import at.fhtw.ai.nn.utils.NeuralNetworkBuilder;
import at.fhtw.ai.nn.utils.Utils;
import test.loader.DigitImage;
import test.loader.DigitImageLoadingService;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

/**
 * - Removed OTSU Algorithm
 * <p>
 * Created On: 29.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class MnistMain {
    public static void main(String[] args) {
        // Configurations:
        // -- 93.80% --
        // Lr: 0.2
        // M: 0.9
        // E: 0.005
        double learningRate = 0.2;
        double momentum = 0.9;
        double meanNetworkError = 0.0005;

        long startTime = System.currentTimeMillis();

        int epochCount = 0;
        int saveCount = 0;

        int numberOfInputNeurons = 28 * 28;
        int numberOfOutputNeurons = 10;
        int numberOfHiddenNeurons = (int) Math.round(Math.sqrt(numberOfInputNeurons * numberOfOutputNeurons));

        System.out.println("Setting up Neural Network...");
        NeuralNetwork neuralNetwork = new NeuralNetworkBuilder()
                .layer("Input Layer", numberOfInputNeurons)
                .layer("Hidden layer", numberOfHiddenNeurons)
                .layer("Output Layer", numberOfOutputNeurons)
                .activationFunction(new Sigmoid())
                .initializer(new XavierInitializer())
                .build();

        BackPropagation backPropagation = new BackPropagationBuilder()
                .learningRate(learningRate)
                .momentum(momentum)
                .neuralNetwork(neuralNetwork)
                .build();

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
                neuralNetwork.input(digitImage.getData());

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
