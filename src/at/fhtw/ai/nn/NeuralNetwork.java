package at.fhtw.ai.nn;

import at.fhtw.ai.nn.activation.ActivationFunction;
import at.fhtw.ai.nn.connect.Connector;
import at.fhtw.ai.nn.initialize.Initializer;
import at.fhtw.ai.nn.initialize.RandomInitializer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * The neural network is an artificial replica of a biological brain. It uses layers, neurons and synapses to compute and predict certain values.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 5429505999185890927L;

    /**
     * Contains all layers of the neural network.
     */
    private List<Layer> layers = new ArrayList<>(8);

    /**
     * Initializer for the neural network.
     */
    private Initializer initializer = new RandomInitializer();

    /**
     * Activation function.
     */
    private ActivationFunction activationFunction;

    /**
     * Layer connection algorithm.
     */
    private Connector connector;


    /**
     * Returns the input layer. This is the first layer in the list of layers in the network.
     *
     * @return Input layer.
     */
    public Layer getInputLayer() {
        if (layers.isEmpty()) {
            return null;
        }
        return layers.get(0);
    }

    /**
     * Returns the output layer. This is the last layer in the list of layers in the network.
     *
     * @return Output layer.
     */
    public Layer getOutputLayer() {
        if (layers.isEmpty()) {
            return null;
        }
        return layers.get(layers.size() - 1);
    }

    /**
     * Sets all the network layers.
     *
     * @param layers Layers.
     */
    public void setLayers(List<Layer> layers) {
        this.layers = layers;
    }

    /**
     * Returns all the network layers.
     *
     * @return Layers
     */
    public List<Layer> getLayers() {
        return layers;
    }


    /**
     * Sets the activation function for all neurons in all layers.
     *
     * @param activationFunction New global activation function.
     */
    public void setActivationFunctions(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * Sets the layer connection algorithm for all layers.
     *
     * @param connector Connector.
     */
    public void setConnectors(Connector connector) {
        this.connector = connector;
    }

    /**
     * Connects all layers in a standard order.
     */
    public void connectLayersInOrder() {
        if (layers.size() <= 1) {
            return;
        }

        if (connector != null) {
            layers.forEach(layer -> layer.setConnector(connector));
        }

        Layer previousLayer = layers.get(0);
        for (int i = 1; i < layers.size(); i++) {
            Layer currentLayer = layers.get(i);
            currentLayer.connectInput(previousLayer);
            previousLayer = currentLayer;
        }
    }

    /**
     * Returns a list which contains all neurons in the neural network.
     *
     * @return All neurons of this network.
     */
    public List<Neuron> getNeurons() {
        List<Neuron> result = new ArrayList<>(512);
        layers.stream().forEach(layer -> result.addAll(layer.getNeurons()));
        return result;
    }

    /**
     * Returns a list which contains all synapses in the neural network.
     *
     * @return All synapses of this network.
     */
    public List<Synapse> getSynapses() {
        List<Synapse> result = new ArrayList<>(1024);
        getNeurons().stream().forEach(neuron -> result.addAll(neuron.getInputSynapses()));
        return result;
    }

    /**
     * Inputs the given values into the input layer.
     *
     * @param inputValues Input values.
     */
    public void input(double... inputValues) {
        if (inputValues.length != getInputLayer().getNeurons().size()) {
            throw new IllegalArgumentException("Number of inputs does not match number of input neurons");
        }
        for (int i = 0; i < inputValues.length; i++) {
            getInputLayer().getNeurons().get(i).value = inputValues[i];
        }
    }

    /**
     * Returns the output of the output layer. This method should only be used after firing the neural network.
     *
     * @return Output values.
     */
    public double[] output() {
        double[] result = new double[getOutputLayer().getNeurons().size()];
        for (int i = 0; i < getOutputLayer().getNeurons().size(); i++) {
            result[i] = getOutputLayer().getNeurons().get(i).value;
        }
        return result;
    }

    /**
     * Resets the fired state for all neurons.
     */
    private void resetNeuronFiredState() {
        layers.forEach(layer -> layer.setNeuronsFired(false));
    }


    /**
     * Fires all neurons in all layers at once.
     */
    public void fire() {
        layers.stream().forEach(layer -> layer.fire());
        resetNeuronFiredState();
    }

    /**
     * Fires only the output layer neurons.
     */
    public void fireOutput() {
        getOutputLayer().fireParallel();
        resetNeuronFiredState();
    }

    /**
     * Sets the neural network initializer.
     *
     * @param initializer Initializer.
     */
    public void setInitializer(Initializer initializer) {
        this.initializer = initializer;
    }

    /**
     * Returns the neural network initializer.
     *
     * @return Initializer.
     */
    public Initializer getInitializer() {
        return initializer;
    }

    /**
     * Initializes the neural network.
     */
    public void initialize() {
        if (activationFunction != null) {
            layers.forEach(layer -> layer.setActivationFunctions(activationFunction));
        }
        initializer.initialize(this);
    }
}
