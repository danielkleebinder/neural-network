package at.fhtw.ai.nn.initialize;

import at.fhtw.ai.nn.NeuralNetwork;

import java.io.Serializable;

/**
 * Initializer are used to set up a neural network.
 * <p>
 * Created On: 30.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public interface Initializer extends Serializable {

    /**
     * Initializes the given neural network.
     *
     * @param neuralNetwork Neural network.
     */
    void initialize(NeuralNetwork neuralNetwork);
}
