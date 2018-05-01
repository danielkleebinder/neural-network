package at.fhtw.ai.nn.utils;

import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.learning.BackPropagation;
import javafx.util.Builder;

/**
 * A useful helper class for building a back propagation learning algorithm.
 * <p>
 * Created On: 30.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class BackPropagationBuilder implements Builder<BackPropagation> {

    /**
     * Resulting back propagation.
     */
    private BackPropagation backPropagation;

    /**
     * Creates a new back propagation builder.
     */
    public BackPropagationBuilder() {
        backPropagation = new BackPropagation();
    }

    /**
     * Sets the learning rate of the back propagation algorithm.
     *
     * @param learningRate Learning rate.
     * @return This back propagation builder.
     */
    public BackPropagationBuilder learningRate(double learningRate) {
        backPropagation.setLearningRate(learningRate);
        return this;
    }

    /**
     * Sets the momentum of the back propagation algorithm.
     *
     * @param momentum Momentum.
     * @return This back propagation builder.
     */
    public BackPropagationBuilder momentum(double momentum) {
        backPropagation.setMomentum(momentum);
        return this;
    }

    /**
     * Sets the neural network.
     *
     * @param neuralNetwork Neural network.
     * @return This back propagation builder.
     */
    public BackPropagationBuilder neuralNetwork(NeuralNetwork neuralNetwork) {
        backPropagation.setNeuralNetwork(neuralNetwork);
        return this;
    }

    @Override
    public BackPropagation build() {
        return backPropagation;
    }
}