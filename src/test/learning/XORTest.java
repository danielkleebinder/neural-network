package test.learning;

import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.activation.Identity;
import at.fhtw.ai.nn.activation.Sigmoid;
import at.fhtw.ai.nn.activation.rectifier.SwishRectifier;
import at.fhtw.ai.nn.initialize.XavierInitializer;
import at.fhtw.ai.nn.learning.BackPropagation;
import at.fhtw.ai.nn.loss.Quadratic;
import at.fhtw.ai.nn.utils.NeuralNetworkBuilder;

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
        int numberOfHiddenNeurons = 5;

        NeuralNetwork neuralNetwork = new NeuralNetworkBuilder()
                .inputLayer("Input Layer", numberOfInputNeurons, new Identity())
                .hiddenLayer("Hidden Layer", numberOfHiddenNeurons, new SwishRectifier())
                .outputLayer("Output Layer", numberOfOutputNeurons, new Sigmoid())
                .initializer(new XavierInitializer())
                .build();

        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setLossFunction(new Quadratic());
        backPropagation.setLearningRate(0.2);
        backPropagation.setMomentum(0.9);
        backPropagation.setMeanSquareError(0.005);
        backPropagation.setNeuralNetwork(neuralNetwork);

        Random rnd = new Random(1);
        int index = 0;
        do {
            int a = rnd.nextInt(2);
            int b = rnd.nextInt(2);
            int c = a ^ b;

            neuralNetwork.input(a, b);

            backPropagation.getDesiredOutputValues().clear();
            backPropagation.getDesiredOutputValues().add((double) c);

            do {
                backPropagation.learn();
            } while (backPropagation.networkError() > backPropagation.getMeanSquareError());
            index++;
        } while (index < 1000);

        System.out.println("Iterations: " + index);
        int correct = 0;
        int total = 1000;
        for (int i = 0; i < total; i++) {
            int a = rnd.nextInt(2);
            int b = rnd.nextInt(2);
            int c = a ^ b;

            neuralNetwork.input(a, b);
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
