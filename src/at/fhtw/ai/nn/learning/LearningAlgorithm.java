package at.fhtw.ai.nn.learning;

import at.fhtw.ai.nn.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public abstract class LearningAlgorithm implements Learnable {

    /**
     * Neural network.
     */
    protected NeuralNetwork neuralNetwork;

    /**
     * Contains all desired output values.
     */
    protected List<Double> desiredOutputValues = new ArrayList<>(16);


    /**
     * Creates a new learning algorithm.
     */
    public LearningAlgorithm() {
    }

    /**
     * Creates the learning algorithm for the given neural network.
     *
     * @param neuralNetwork Neural network.
     */
    public LearningAlgorithm(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    /**
     * Sets the neural network for learning.
     *
     * @param neuralNetwork Neural network.
     */
    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    /**
     * Returns the neural network.
     *
     * @return Neural network.
     */
    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }


    /**
     * Sets the desired output values.
     *
     * @param desiredOutputValues Desired output values.
     */
    public void setDesiredOutputValues(List<Double> desiredOutputValues) {
        this.desiredOutputValues = desiredOutputValues;
    }

    /**
     * Returns the desired output values.
     *
     * @return Desired output values.
     */
    public List<Double> getDesiredOutputValues() {
        return desiredOutputValues;
    }
}
