package test.learning;

import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.activation.rectifier.ExponentialRectifier;
import at.fhtw.ai.nn.activation.rectifier.ParametricRectifier;
import at.fhtw.ai.nn.connect.DenseConnector;
import at.fhtw.ai.nn.initialize.XavierInitializer;
import at.fhtw.ai.nn.learning.BackPropagation;
import at.fhtw.ai.nn.loss.Quadratic;
import at.fhtw.ai.nn.regularization.L2;
import at.fhtw.ai.nn.utils.NeuralNetworkBuilder;

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
        NeuralNetwork neuralNetwork = new NeuralNetworkBuilder()
                .layer("Input Layer", 2)
                .layer("Hidden Layer 1", 4)
                .layer("Hidden Layer 2", 3)
                .layer("Output Layer", 1)
                .activationFunction(new ExponentialRectifier())
                .connector(new DenseConnector())
                .initializer(new XavierInitializer())
                .build();

        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setLossFunction(new Quadratic());
        backPropagation.setRegularization(new L2());
        backPropagation.setLearningRate(0.2);
        backPropagation.setMomentum(0.9);
        backPropagation.setMeanSquareError(0.05);                     // Change to 0.005 later
        backPropagation.setNeuralNetwork(neuralNetwork);

        Random rnd = new Random();
        do {
            int a = rnd.nextInt(16);
            int b = rnd.nextInt(16);
            int c = a + b;

            neuralNetwork.getInputLayer().getNeurons().get(0).value = a;
            neuralNetwork.getInputLayer().getNeurons().get(1).value = b;

            backPropagation.getDesiredOutputValues().clear();
            backPropagation.getDesiredOutputValues().add((double) c);
            backPropagation.learn();
        } while (backPropagation.networkError() > backPropagation.getMeanSquareError());

        System.out.println(neuralNetwork.getOutputLayer().getNeurons().get(0).bias);
        run(neuralNetwork, 1, 2);
        run(neuralNetwork, 2, 1);
        run(neuralNetwork, 5, 7);
    }

    private static void run(NeuralNetwork nn, double a, double b) {
        nn.getInputLayer().getNeurons().get(0).value = a;
        nn.getInputLayer().getNeurons().get(0).value = b;
        nn.fireOutput();
        System.out.println(a + " + " + b + " = " + nn.output()[0] + " (Expected: " + (a + b) + ")");
    }
}
