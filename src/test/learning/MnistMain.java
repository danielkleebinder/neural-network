package test.learning;

import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.activation.Identity;
import at.fhtw.ai.nn.activation.layer.Softmax;
import at.fhtw.ai.nn.activation.rectifier.ExponentialRectifier;
import at.fhtw.ai.nn.activation.rectifier.SwishRectifier;
import at.fhtw.ai.nn.connect.DenseConnector;
import at.fhtw.ai.nn.initialize.XavierInitializer;
import at.fhtw.ai.nn.learning.BackPropagation;
import at.fhtw.ai.nn.loss.CrossEntropy;
import at.fhtw.ai.nn.normalization.MinMax;
import at.fhtw.ai.nn.regularization.Dropout;
import at.fhtw.ai.nn.utils.BackPropagationBuilder;
import at.fhtw.ai.nn.utils.NeuralNetworkBuilder;
import at.fhtw.ai.nn.utils.Utils;
import test.loader.GrayImage;
import test.loader.MNISTImageLoadingService;
import test.loader.MnistImage;

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
                .layer("Input Layer", numberOfInputNeurons, new Identity())
                .layer("Hidden layer (Swish)", numberOfHiddenNeurons, new SwishRectifier())
                .layer("Hidden layer (ELU)", numberOfHiddenNeurons / 2, new ExponentialRectifier())
                .layer("Output Layer", numberOfOutputNeurons, new Softmax())
                .connector(new DenseConnector())
                .initializer(new XavierInitializer())
                .normalization(new MinMax())
                .build();

        System.out.println("   Number of layers: " + neuralNetwork.getLayers().size());
        System.out.println("   Number of neurons: " + neuralNetwork.getNeurons().size());
        System.out.println("   Number of synapses: " + neuralNetwork.getSynapses().size());

        BackPropagation backPropagation = new BackPropagationBuilder()
                .regularization(new Dropout())
                .lossFunction(new CrossEntropy())
                .learningRate(learningRate)
                .momentum(momentum)
                .neuralNetwork(neuralNetwork)
                .build();

        System.out.println();
        System.out.println("Loading Train Images...");
        MNISTImageLoadingService dilsTrainData = new MNISTImageLoadingService(
                "C:/Users/Daniel/Desktop/train/train-labels-idx1-ubyte.dat",
                "C:/Users/Daniel/Desktop/train/train-images-idx3-ubyte.dat"
        );

        List<MnistImage> trainImages = null;
        try {
            trainImages = dilsTrainData.loadMNISTImages(new GrayImage.Factory());
            Collections.shuffle(trainImages);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Pre-Initialize desired output values for back propagation
        for (int i = 0; i < neuralNetwork.getOutputLayer().getNeurons().size(); i++) {
            backPropagation.getDesiredOutputValues().add(0.0);
        }

        // Do learning here
        System.out.println();
        System.out.println("Start Learning Process...");
        int epochIterations = 0;
        double epochError = 0.0;
        double averageNetworkError;
        do {
            Collections.shuffle(trainImages);
            for (MnistImage mnistImage : trainImages) {
                // Set neural network input parameters
                neuralNetwork.input(mnistImage.getData());

                // Set desired output values
                for (int i = 0; i < backPropagation.getDesiredOutputValues().size(); i++) {
                    backPropagation.getDesiredOutputValues().set(i, 0.0);
                }
                backPropagation.getDesiredOutputValues().set(mnistImage.getLabel(), 1.0);

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
