package at.fhtw.ai.nn.regularization;

import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.utils.AtomicDouble;

/**
 * The L1 regularization technique, also known as Lasse Regression, is a commonly used regularization method for weights
 * in neural networks.<br>
 * <code>C = (lambda / 2) * sum(abs(w))</code>
 * <p>
 * Created On: 01.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class L1 extends AbstractRegularization {
    private static final long serialVersionUID = -9100303352979670830L;

    @Override
    public double compute(NeuralNetwork neuralNetwork) {
        AtomicDouble sumWeights = new AtomicDouble(0.0);
        neuralNetwork.getSynapses().stream().forEach(synapse -> sumWeights.value += Math.abs(synapse.weight));
        return sumWeights.value * (lambda / 2.0);
    }
}