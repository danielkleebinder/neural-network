package test;

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
 * TrainMain program entry point.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Train2Main {
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
        backPropagation.setLearningRate(0.05);
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
            Collections.shuffle(trainImages);
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < 10; i++) {
            backPropagation.getDesiredOutputValues().add(0.0);
        }

        System.out.println("Learning in Progress...");
        double networkError;
        for (int i = 0; i < trainImages.size(); i++) {
            DigitImage digitImage = trainImages.get(i);
            for (int j = 0; j < digitImage.getData().length; j++) {
                inputLayer.getNeurons().get(j).setValue(digitImage.getData()[j]);
                inputLayer.getNeurons().get(j).getBias().setValue(0.0);
                inputLayer.getNeurons().get(j).getBias().setWeight(0.0);
            }
            for (int j = 0; j < 10; j++) {
                backPropagation.getDesiredOutputValues().set(j, 0.0);
            }
            backPropagation.getDesiredOutputValues().set(digitImage.getLabel(), 1.0);

            int iterations = 0;
            do {
                iterations++;
                backPropagation.learn();
                networkError = backPropagation.networkError();
            } while (networkError > backPropagation.getMeanSquareError() && iterations < 512);
            if ((i + 1) % 500 == 0) {
                System.out.println("     -> Quick saving Neural Network ...");
                save(neuralNetwork);
                System.out.println("     -> Successfully Saved! Pausing for 10 Seconds ...");
                try {
                    Thread.sleep(10000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            int progress = (int) Math.round(((i + 1) / (double) trainImages.size()) * 100.0);
            System.out.printf("[%03d%%] - %05d of %05d done (Square Error: %f, Iterations: %d)\n", progress, (i + 1), trainImages.size(), networkError, iterations);
        }
        System.out.println("Learning Done!");

        System.out.println("Now Serializing Neural Network...");
        save(neuralNetwork);
        System.out.println("Neural Network Successfully Serialized!");
    }

    private static void save(NeuralNetwork neuralNetwork) {
        try {
            Utils.serialize(neuralNetwork, "C:/Users/Daniel/Desktop/NeuralNetwork_M23.dat");
        } catch (IOException e) {
            e.printStackTrace();
        }/// TODO: Change bias weight back to 1.0
    }
}
