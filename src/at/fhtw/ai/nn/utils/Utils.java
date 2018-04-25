package at.fhtw.ai.nn.utils;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.Neuron;

import java.io.*;
import java.util.Random;
import java.util.UUID;

/**
 * A utility class which contains a lot of useful methods for neural networks.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Utils {

    /**
     * Global random instance.
     */
    private static final Random RND = new Random();

    /**
     * Nobody is allowed to create an instance of this class!
     */
    private Utils() {
    }

    /**
     * Creates an empty layer.
     *
     * @return Layer.
     */
    public static Layer createLayer() {
        return createLayer(null);
    }

    /**
     * Creates a new empty layer with the given name.
     *
     * @param name Name.
     * @return Layer.
     */
    public static Layer createLayer(String name) {
        return createLayer(name, 0);
    }

    /**
     * Creates a new layer with the given amount of neurons.
     *
     * @param neurons Number of neurons.
     * @return Layer.
     */
    public static Layer createLayer(int neurons) {
        return createLayer(null, neurons);
    }

    /**
     * Creates a new layer with the given name and amount of neurons.
     *
     * @param name    Name.
     * @param neurons Number of neurons.
     * @return Layer.
     */
    public static Layer createLayer(String name, int neurons) {
        Layer result = new Layer();
        if (name == null) {
            result.setName("Layer-" + UUID.randomUUID().toString());
        } else {
            result.setName(name);
        }
        for (int i = 0; i < neurons; i++) {
            Neuron neuron = createNeuron();
            neuron.getBias().setValue(randomBias());
            neuron.getBias().setWeight(randomWeight());
            result.getNeurons().add(neuron);
        }
        return result;
    }

    /**
     * Creates a new neuron.
     *
     * @return Neuron.
     */
    public static Neuron createNeuron() {
        return new Neuron();
    }

    /**
     * Returns a random weight between -1 and 1.
     *
     * @return Random weight.
     */
    public static double randomWeight() {
        return RND.nextDouble() * 2.0 - 1.0;
    }

    /**
     * Returns a random bias value (either -1 or 1).
     *
     * @return Random bias.
     */
    public static double randomBias() {
        return RND.nextBoolean() ? -1.0 : 1.0;
    }

    /**
     * Serializes the given neural network.
     *
     * @param neuralNetwork Neural network.
     * @param outputFile    Output file name.
     * @throws IOException May occur during serialization process.
     */
    public static void serialize(NeuralNetwork neuralNetwork, String outputFile) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(outputFile))) {
            oos.writeObject(neuralNetwork);
            oos.flush();
        }
    }

    /**
     * Deserializes the given input file into a neural network.
     *
     * @param inputFile Input file.
     * @return Neural network.
     * @throws IOException            May occur during deserialization process.
     * @throws ClassNotFoundException May occur during loading process.
     */
    public static NeuralNetwork deserialize(String inputFile) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(inputFile))) {
            return (NeuralNetwork) ois.readObject();
        }
    }
}
