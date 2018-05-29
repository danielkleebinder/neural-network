package at.fhtw.ai.nn.utils;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.activation.ActivationFunction;
import at.fhtw.ai.nn.connect.Connector;
import at.fhtw.ai.nn.initialize.Initializer;
import at.fhtw.ai.nn.normalization.Normalization;
import javafx.util.Builder;

/**
 * A useful helper class for building a neural network.
 * <p>
 * Created On: 30.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class NeuralNetworkBuilder implements Builder<NeuralNetwork> {

    /**
     * Resulting neural network.
     */
    private NeuralNetwork neuralNetwork;
    private Layer inputLayer;
    private Layer outputLayer;

    /**
     * Creates a new neural network builder.
     */
    public NeuralNetworkBuilder() {
        neuralNetwork = new NeuralNetwork();
    }

    /**
     * Sets the initializer of the neural network.
     *
     * @param initializer Initializer.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder initializer(Initializer initializer) {
        neuralNetwork.setInitializer(initializer);
        return this;
    }

    /**
     * Adds a new layer with the given number of neurons to the network.
     *
     * @param neurons Number of neurons for this layer.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder layer(int neurons) {
        return layer(null, neurons);
    }

    /**
     * Adds a new layer with the given name and number of neurons to the network.
     *
     * @param name    Layer name.
     * @param neurons Number of neurons for this layer.
     * @return A neural network layer builder.
     */
    public NeuralNetworkBuilder layer(String name, int neurons) {
        return layer(name, neurons, (ActivationFunction) null);
    }

    /**
     * Adds a new layer with the given name, number of neurons and activation function to the network.
     *
     * @param name               Layer name.
     * @param neurons            Number of neurons for this layer.
     * @param activationFunction Activation function.
     * @return A neural network layer builder.
     */
    public NeuralNetworkBuilder layer(String name, int neurons, ActivationFunction activationFunction) {
        return layer(name, neurons, activationFunction, null);
    }

    /**
     * Adds a new layer with the given name, number of neurons and connector to the network.
     *
     * @param name      Layer name.
     * @param neurons   Number of neurons for this layer.
     * @param connector Connector.
     * @return A neural network layer builder.
     */
    public NeuralNetworkBuilder layer(String name, int neurons, Connector connector) {
        return layer(name, neurons, null, connector);
    }

    /**
     * Adds a new layer with the given name, number of neurons, activation function and connector to the network.
     *
     * @param name               Layer name.
     * @param neurons            Number of neurons for this layer.
     * @param activationFunction Activation function.
     * @param connector          Connector.
     * @return A neural network layer builder.
     */
    public NeuralNetworkBuilder layer(String name, int neurons, ActivationFunction activationFunction, Connector connector) {
        Layer layer = Utils.createLayer(name, neurons);
        if (activationFunction != null) {
            layer.setActivationFunctions(activationFunction);
        }
        if (connector != null) {
            layer.setConnector(connector);
        }
        neuralNetwork.getLayers().add(layer);
        return this;
    }

    /**
     * Adds a new hidden layer with the given number of neurons to the network. Using this method is equivalent to
     * <code>layer(int neurons)</code>.
     *
     * @param neurons Number of neurons for this layer.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder hiddenLayer(int neurons) {
        return layer(neurons);
    }

    /**
     * Adds a new hidden layer with the given name and number of neurons to the network. Using this method is equivalent
     * to <code>layer(String name, int neurons)</code>.
     *
     * @param name    Layer name.
     * @param neurons Number of neurons for this layer.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder hiddenLayer(String name, int neurons) {
        return layer(name, neurons);
    }

    /**
     * Adds a new hidden layer with the given name, number of neurons and activation function to the network. Using this
     * method is equivalent to <code>layer(String name, int neurons, ActivationFunction activationFunction)</code>.
     *
     * @param name               Layer name.
     * @param neurons            Number of neurons for this layer.
     * @param activationFunction Activation function.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder hiddenLayer(String name, int neurons, ActivationFunction activationFunction) {
        return layer(name, neurons, activationFunction);
    }


    /**
     * Sets the input layer of the neural network.
     *
     * @param neurons Number of neurons for this layer.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder inputLayer(int neurons) {
        return inputLayer(null, neurons);
    }

    /**
     * Sets the input layer of the neural network. If the input layer is not explicitly specified, the first layer of
     * the network will be used as input layer.
     *
     * @param name    Layer name.
     * @param neurons Number of neurons for this layer.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder inputLayer(String name, int neurons) {
        inputLayer = Utils.createLayer(name, neurons);
        return this;
    }

    /**
     * Sets the input layer of the neural network. If the input layer is not explicitly specified, the first layer of
     * the network will be used as input layer.
     *
     * @param name               Layer name.
     * @param neurons            Number of neurons for this layer.
     * @param activationFunction Activation function.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder inputLayer(String name, int neurons, ActivationFunction activationFunction) {
        inputLayer = Utils.createLayer(name, neurons);
        inputLayer.setActivationFunctions(activationFunction);
        return this;
    }

    /**
     * Sets the output layer of the neural network.
     *
     * @param neurons Number of neurons for this layer.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder outputLayer(int neurons) {
        return outputLayer(null, neurons);
    }

    /**
     * Sets the output layer of the neural network. If the output layer is not explicitly specified, the last layer of
     * the network will be used as output layer.
     *
     * @param name    Layer name.
     * @param neurons Number of neurons for this layer.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder outputLayer(String name, int neurons) {
        outputLayer = Utils.createLayer(name, neurons);
        return this;
    }

    /**
     * Sets the output layer of the neural network. If the output layer is not explicitly specified, the last layer of
     * the network will be used as output layer.
     *
     * @param name               Layer name.
     * @param neurons            Number of neurons for this layer.
     * @param activationFunction Activation function.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder outputLayer(String name, int neurons, ActivationFunction activationFunction) {
        outputLayer = Utils.createLayer(name, neurons);
        outputLayer.setActivationFunctions(activationFunction);
        return this;
    }

    /**
     * Sets the activation function of the neural network. This method call will overwrite all activation functions in
     * all neurons and layers of this network.
     *
     * @param activationFunction Activation function.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder activationFunction(ActivationFunction activationFunction) {
        neuralNetwork.setActivationFunctions(activationFunction);
        return this;
    }

    /**
     * Sets the connector for the neural network.
     *
     * @param connector Connector.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder connector(Connector connector) {
        neuralNetwork.setConnectors(connector);
        return this;
    }

    /**
     * Sets the normalization algorithm for all input values.
     *
     * @param normalization Normalization algorithm.
     * @return This neural network builder.
     */
    public NeuralNetworkBuilder normalization(Normalization normalization) {
        neuralNetwork.setNormalization(normalization);
        return this;
    }

    @Override
    public NeuralNetwork build() {
        if (inputLayer != null) {
            neuralNetwork.getLayers().add(0, inputLayer);
        }
        if (outputLayer != null) {
            neuralNetwork.getLayers().add(outputLayer);
        }
        neuralNetwork.connectLayersInOrder();
        neuralNetwork.initialize();
        return neuralNetwork;
    }
}