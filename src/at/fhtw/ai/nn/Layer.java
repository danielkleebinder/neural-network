package at.fhtw.ai.nn;

import at.fhtw.ai.nn.activation.ActivationFunction;
import at.fhtw.ai.nn.activation.layer.LayerActivationFunction;
import at.fhtw.ai.nn.connect.Connector;
import at.fhtw.ai.nn.connect.DenseConnector;

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
     * Layer connection algorithm.
     */
    private Connector connector = new DenseConnector();

    /**
     * Layer activation function.
     */
    private LayerActivationFunction layerActivationFunction;

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
     * Sets the layer connection algorithm.
     *
     * @param connector Connector.
     */
    public void setConnector(Connector connector) {
        this.connector = connector;
    }

    /**
     * Returns the layer connection algorithm.
     *
     * @return Connector.
     */
    public Connector getConnector() {
        return connector;
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
        if (activationFunction instanceof LayerActivationFunction) {
            layerActivationFunction = (LayerActivationFunction) activationFunction;
        }
        neurons.forEach(neuron -> neuron.setActivationFunction(activationFunction));
    }

    /**
     * Returns the layer based activation function if one is used.
     *
     * @return Layer based activation function.
     */
    public LayerActivationFunction getLayerActivationFunction() {
        return layerActivationFunction;
    }

    /**
     * Connects the given input layer to this layer.
     *
     * @param inputLayer Input layer.
     */
    public void connectInput(Layer inputLayer) {
        connector.connect(inputLayer, this);
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
        neurons.forEach(neuron -> neuron.preCompute(false));
        if (layerActivationFunction != null) {
            layerActivationFunction.initialize(this);
        }
        neurons.forEach(neuron -> neuron.fire(false));
    }

    /**
     * Fires all neurons in parallel.
     */
    public void fireParallel() {
        neurons.parallelStream().forEach(neuron -> neuron.preCompute(false));
        if (layerActivationFunction != null) {
            layerActivationFunction.initialize(this);
        }
        neurons.parallelStream().forEach(neuron -> neuron.fire(false));
    }

    @Override
    public String toString() {
        return name;
    }
}