package test;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.activation.Identity;
import at.fhtw.ai.nn.activation.Sigmoid;
import at.fhtw.ai.nn.learning.BackPropagation;
import at.fhtw.ai.nn.utils.Utils;

import java.util.Random;

/**
 * TrainMain entry point for XOR neuron network test.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class XORTest {
    public static void main(String[] args) {
        int numberOfInputNeurons = 2;
        int numberOfOutputNeurons = 1;
        int numberOfHiddenNeurons = 2;

        Layer inputLayer = Utils.createLayer("Input Layer", numberOfInputNeurons);
        Layer outputLayer = Utils.createLayer("Output Layer", numberOfOutputNeurons);
        Layer hiddenLayer = Utils.createLayer("Hidden Layer", numberOfHiddenNeurons);

        System.out.println(inputLayer + ": " + inputLayer.getNeurons());
        System.out.println(hiddenLayer + ": " + hiddenLayer.getNeurons());
        System.out.println(outputLayer + ": " + outputLayer.getNeurons());

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

        Random rnd = new Random();
        int index = 0;
        do {
            int a = rnd.nextInt(2);
            int b = rnd.nextInt(2);
            int c = a ^ b;

            inputLayer.getNeurons().get(0).getBias().setValue(a);
            inputLayer.getNeurons().get(1).getBias().setValue(b);

            backPropagation.getDesiredOutputValues().clear();
            backPropagation.getDesiredOutputValues().add((double) c);
            backPropagation.learn();
            index++;
        } while (backPropagation.networkError() > backPropagation.getMeanSquareError() || index < 100);

        System.out.println("Iterations: " + index);
        int correct = 0;
        int total = 1000;
        for (int i = 0; i < total; i++) {
            int a = rnd.nextInt(2);
            int b = rnd.nextInt(2);
            int c = a ^ b;

            inputLayer.getNeurons().get(0).getBias().setValue(a);
            inputLayer.getNeurons().get(1).getBias().setValue(b);

            neuralNetwork.fireOutput();

            double r = neuralNetwork.getOutputLayer().getNeurons().get(0).getValue();

            if (c == (int) Math.round(r)) {
                correct++;
            }
        }
        System.out.println("Total: " + total);
        System.out.println("Correct: " + correct);
        System.out.println("Wrong: " + (total - correct));
        System.out.println("Accuracy: " + (correct / (double) total * 100.0) + "%");
    }
}
