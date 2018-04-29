package test.learning;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.Synapse;
import at.fhtw.ai.nn.activation.Sigmoid;
import at.fhtw.ai.nn.learning.BackPropagation;
import at.fhtw.ai.nn.utils.Utils;
import test.loader.DigitImage;
import test.loader.DigitImageLoadingService;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

/**
 * TrainMain program entry point.
 * <p>
 * Created On: 24.04.2018
 * <p>
 * CHANGED:
 * - Bias only 1 (not 1 OR -1)
 * - Neurons are over saturated (value too high for activation function)
 * - Multithreading causes race conditions and invalidates neuron outputs
 * - Concurrency on value in neuron
 * - Only fire once
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Train2Main {
    private static int saveCount = 0;
    private static final int epochs = 1;                     // 20, 256
    private static final int quickSaving = 10000;               // 5000
    private static final double learningRate = 0.2;            // 0.01, 0.2, 0.03
    private static final double momentum = 0.9;
    private static final double meanSquareError = 0.05;            // 0.005

    public static void main(String[] args) {
        int numberOfInputNeurons = 28 * 28;
        int numberOfOutputNeurons = 10;
        int numberOfHiddenNeurons = (int) Math.round(Math.sqrt(numberOfInputNeurons * numberOfOutputNeurons));

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
        backPropagation.setMeanSquareError(meanSquareError);
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

        for (int i = 0; i < 10; i++) {
            backPropagation.getDesiredOutputValues().add(0.0);
        }

        System.out.println("Learning in Progress...");
        double averageIterations = epochs;
        double totalIterations = 0.0;
        double networkError;
        double totalNetworkError = 0.0;
        double epochCount = 0;
        long startTime = System.currentTimeMillis();
        do {
            Collections.shuffle(trainImages);
            totalIterations = 0.0;
            totalNetworkError = 0.0;
            for (int i = 0; i < trainImages.size(); i++) {
                DigitImage digitImage = trainImages.get(i);
                for (int j = 0; j < digitImage.getData().length; j++) {
                    inputLayer.getNeurons().get(j).setValue(digitImage.getData()[j]);
                }
                for (int j = 0; j < 10; j++) {
                    backPropagation.getDesiredOutputValues().set(j, 0.0);
                }
                backPropagation.getDesiredOutputValues().set(digitImage.getLabel(), 1.0);
                //int iterations = 0;
                //do {
                //iterations++;
                    backPropagation.learn();
                    networkError = backPropagation.networkError();
                    totalNetworkError += networkError;
                    totalIterations++;
                //} while (networkError > backPropagation.getMeanSquareError() && iterations < 512);
                //totalIterations += iterations;

                if ((i + 1) % quickSaving == 0) {
                    System.out.println("     -> Milliseconds since start: " + (System.currentTimeMillis() - startTime) + " ms");
                    System.out.println("     -> Quick saving Neural Network ...");
                    save(neuralNetwork, (int) epochCount);
                    System.out.println("     -> Successfully Saved! Pausing for 5 Seconds ...");
                    //totalIterations = 0.0;
                    try {
                        Thread.sleep(5000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }

                //int progress = (int) Math.round(((i + 1) / (double) trainImages.size()) * 100.0);
                //System.out.printf("[%03d%%] - %05d of %05d done (Square Error: %f, Iterations: %d)\n", progress, (i + 1), trainImages.size(), networkError, iterations);
            }
            epochCount++;
            System.out.println("Epoch-" + (epochCount - 1) + " Finished with Network Error of " + (totalNetworkError / totalIterations));
            saveCount = 0;
        } while ((totalNetworkError / totalIterations) > backPropagation.getMeanSquareError());
        System.out.println("Learning Done!");

        System.out.println("Now Serializing Neural Network...");
        save(neuralNetwork, (int) epochCount);
        System.out.println("Neural Network Successfully Serialized!");
    }

    private static void save(NeuralNetwork neuralNetwork, int epochCount) {
        try {
            Utils.serialize(neuralNetwork, "C:/Users/Daniel/Desktop/pl3/NeuralNetwork_E" + epochCount + "_C" + saveCount + ".dat");
            saveCount++;
        } catch (IOException e) {
            e.printStackTrace();
        }/// TODO: Change bias weight back to 1.0
    }
}
