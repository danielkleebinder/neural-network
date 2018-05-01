package at.fhtw.ai.nn.initialize;

import at.fhtw.ai.nn.NeuralNetwork;

import java.util.Random;

/**
 * The xavier initializer is a special kind of neural network initializer which optimizes the randomized weights
 * at the beginning for an optimal learning process.
 * <p>
 * Created On: 30.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class XavierInitializer implements Initializer {
    private static final long serialVersionUID = 1931237022211839558L;

    /**
     * Random object for initialization.
     */
    private Random rnd;

    /**
     * Creates a new xavier initializer with the current system time in milliseconds as seed.
     */
    public XavierInitializer() {
        this(System.currentTimeMillis());
    }

    /**
     * Creates a new xavier initializer with the given seed.
     *
     * @param seed Seed.
     */
    public XavierInitializer(long seed) {
        rnd = new Random(seed);
    }

    @Override
    public void initialize(NeuralNetwork neuralNetwork) {
        int inputNeurons = neuralNetwork.getInputLayer().getNeurons().size();
        int outputNeurons = neuralNetwork.getOutputLayer().getNeurons().size();

        // Use average neuron count to preserve back propagation signal
        double nAvg = (inputNeurons + outputNeurons) / 2.0;
        double variance = 1.0 / nAvg;
        double standardDeviation = Math.sqrt(variance);

        // Initialize weights
        neuralNetwork.getNeurons().stream().forEach(neuron -> {
            neuron.bias.value = randomBias();
            neuron.bias.weight = xavierWeight(standardDeviation);
            neuron.value = 0.0;
        });
        neuralNetwork.getSynapses().stream().forEach(synapse -> {
            synapse.weight = xavierWeight(standardDeviation);
            synapse.change = 0.0;
        });
    }

    /**
     * Returns a random xavier weight.
     *
     * @param standardDeviation Desired standard deviation.
     * @return Xavier weight.
     */
    private double xavierWeight(double standardDeviation) {
        return rnd.nextGaussian() * standardDeviation;
    }

    /**
     * Returns a random bias value (either -1 or 1).
     *
     * @return Random bias.
     */
    private double randomBias() {
        return rnd.nextBoolean() ? -1.0 : 1.0;
    }
}
