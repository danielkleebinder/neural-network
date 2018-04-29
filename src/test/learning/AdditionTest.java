package test.learning;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.activation.Identity;
import at.fhtw.ai.nn.learning.BackPropagation;
import at.fhtw.ai.nn.utils.Utils;

import java.util.Random;

/**
 * TrainMain entry point for addition neuron network test.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class AdditionTest {
    public static void main(String[] args) {
        int numberOfInputNeurons = 2;
        int numberOfOutputNeurons = 1;
        int numberOfHiddenNeurons = 3;

        Layer inputLayer = Utils.createLayer(numberOfInputNeurons);
        Layer outputLayer = Utils.createLayer(numberOfOutputNeurons);
        Layer hiddenLayer = Utils.createLayer(numberOfHiddenNeurons);

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.getLayers().add(inputLayer);
        neuralNetwork.getLayers().add(hiddenLayer);
        neuralNetwork.getLayers().add(outputLayer);
        neuralNetwork.setActivationFunctions(new Identity());
        neuralNetwork.connectLayersInOrder();

        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setLearningRate(0.2);
        backPropagation.setMomentum(0.9);
        backPropagation.setMeanSquareError(0.05);                     // Change to 0.005 later
        backPropagation.setNeuralNetwork(neuralNetwork);

        Random rnd = new Random();
        do {
            int a = rnd.nextInt(16);
            int b = rnd.nextInt(16);
            int c = a + b;

            inputLayer.getNeurons().get(0).getBias().setValue(a);
            inputLayer.getNeurons().get(1).getBias().setValue(b);

            backPropagation.getDesiredOutputValues().clear();
            backPropagation.getDesiredOutputValues().add((double) c);
            backPropagation.learn();
            System.out.println();
            System.out.println(backPropagation.networkError());
        } while (backPropagation.networkError() > backPropagation.getMeanSquareError());
    }
}
