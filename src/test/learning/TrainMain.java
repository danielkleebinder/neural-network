package test.learning;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.activation.Sigmoid;
import at.fhtw.ai.nn.learning.BackPropagation;
import at.fhtw.ai.nn.utils.Utils;
import test.loader.DigitImage;
import test.loader.DigitImageLoadingService;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * TrainMain program entry point.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class TrainMain {
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
        backPropagation.setLearningRate(0.2);
        backPropagation.setMomentum(0.9);
        backPropagation.setMeanSquareError(0.005);                     // Change to 0.005 later
        backPropagation.setNeuralNetwork(neuralNetwork);

        System.out.println("Loading Train Images...");
        DigitImageLoadingService dilsTrainData = new DigitImageLoadingService(
                "C:/Users/Daniel/Desktop/train/train-labels-idx1-ubyte.dat",
                "C:/Users/Daniel/Desktop/train/train-images-idx3-ubyte.dat"
        );

        List<DigitImage> trainImages = null;
        try {
            trainImages = dilsTrainData.loadDigitImages();
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < 10; i++) {
            backPropagation.getDesiredOutputValues().add(0.0);
        }

        System.out.println("Learning in Progress...");
        int iterations = 0;
        int average = 1000;
        double sum = 0.0;
        double networkError;
        double ne = 1.0;
        Random rnd = new Random();
        do {
            iterations++;

            DigitImage digitImage = trainImages.get(rnd.nextInt(trainImages.size()));

            for (int i = 0; i < digitImage.getData().length; i++) {
                inputLayer.getNeurons().get(i).setValue(digitImage.getData()[i]);
            }

            for (int i = 0; i < 10; i++) {
                backPropagation.getDesiredOutputValues().set(i, 0.0);
            }
            backPropagation.getDesiredOutputValues().set(digitImage.getLabel(), 1.0);
            backPropagation.learn();

            networkError = backPropagation.networkError();
            sum += networkError;
            if (iterations % average == 0) {
                ne = sum / (double) average;
                System.out.println("  -> Iteration-" + iterations + ": " + (sum / (double) average));
                sum = 0.0;
            }
            if (iterations % (average * 30) == 0) {
                System.out.println("     -> Quick saving Neural Network ...");
                save(neuralNetwork);
                System.out.println("     -> Successfully Saved! Pausing for 30 Seconds ...");
                try {
                    Thread.sleep(30000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        } while (networkError > backPropagation.getMeanSquareError() || ne > backPropagation.getMeanSquareError());
        System.out.println("Learning Done! (Iterations: " + iterations + ")");

        System.out.println("Now Serializing Neural Network...");
        save(neuralNetwork);
        System.out.println("Neural Network Successfully Serialized!");
    }

    private static void save(NeuralNetwork neuralNetwork) {
        try {
            Utils.serialize(neuralNetwork, "C:/Users/Daniel/Desktop/NeuralNetwork.dat");
        } catch (IOException e) {
            e.printStackTrace();
        }/// TODO: Change bias weight back to 1.0
    }
}
