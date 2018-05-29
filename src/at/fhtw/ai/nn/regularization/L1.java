package at.fhtw.ai.nn.regularization;

import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.Synapse;

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
    private static final long serialVersionUID = 8100184290977301389L;

    /**
     * Creates a new L1 regularization.
     */
    public L1() {
        this(1e-5);
    }

    /**
     * Creates a new L1 regularization with the given lambda.
     *
     * @param lambda Lambda.
     */
    public L1(double lambda) {
        this.lambda = lambda;
    }

    @Override
    public double compute(Synapse synapse) {
        return lambda * Math.signum(synapse.weight);
    }

    @Override
    public double compute(Neuron neuron) {
        return 1.0;
    }
}