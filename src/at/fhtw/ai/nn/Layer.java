package at.fhtw.ai.nn;

import at.fhtw.ai.nn.activation.ActivationFunction;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * A neural network consists of multiple, visible and hidden layers which contain the neurons of the network.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Layer implements Serializable {
    private static final long serialVersionUID = -8677958591398475538L;

    /**
     * Layer name.
     */
    private String name;

    /**
     * Contains all neurons of the layer.
     */
    private List<Neuron> neurons = new ArrayList<>(16);

    /**
     * Sets the layer name.
     *
     * @param name Layer name.
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the layer name.
     *
     * @return Layer name.
     */
    public String getName() {
        return name;
    }

    /**
     * Sets all neurons of the layer.
     *
     * @param neurons Neurons.
     */
    public void setNeurons(List<Neuron> neurons) {
        this.neurons = neurons;
    }

    /**
     * Returns all neurons of the layer.
     *
     * @return Neurons.
     */
    public List<Neuron> getNeurons() {
        return neurons;
    }

    /**
     * Sets the given activation function as function for all neurons in this layer.
     *
     * @param activationFunction Activation function.
     */
    public void setActivationFunctions(ActivationFunction activationFunction) {
        neurons.forEach(neuron -> neuron.setActivationFunction(activationFunction));
    }

    /**
     * Connects the given input layer to this layer.
     *
     * @param inputLayer Input layer.
     */
    public void connectInput(Layer inputLayer) {
        inputLayer.getNeurons().forEach(neuron -> {
            neurons.forEach(current -> current.connectInput(neuron));
        });
    }

    /**
     * Sets all neurons to the given fired state.
     *
     * @param fired True if fired, otherwise false.
     */
    public void setNeuronsFired(boolean fired) {
        neurons.forEach(neuron -> neuron.setFired(fired));
    }

    /**
     * Fires all neurons at once.
     */
    public void fire() {
        neurons.forEach(neuron -> neuron.fire(false));
    }

    /**
     * Fires all neurons in parallel.
     */
    public void fireParallel() {
        neurons.parallelStream().forEach(neuron -> neuron.fire(false));
    }

    @Override
    public String toString() {
        return name;
    }
}