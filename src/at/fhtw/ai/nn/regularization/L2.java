package at.fhtw.ai.nn.regularization;

import at.fhtw.ai.nn.Synapse;

/**
 * The L2 regularization technique is an often used squared ridge regression method for adjusting weights in the neural
 * network to prevent over fitting.<br>
 * <code>C = (lambda / 2) * sum(w^2)</code>
 * <p>
 * Created On: 01.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class L2 extends AbstractRegularization {
    private static final long serialVersionUID = 4140883881503525322L;

    /**
     * Creates a new L2 regularization.
     */
    public L2() {
        this(1e-5);
    }

    /**
     * Creates a new L2 regularization with the given lambda.
     *
     * @param lambda Lambda.
     */
    public L2(double lambda) {
        this.lambda = lambda;
    }

    @Override
    public double compute(Synapse synapse) {
        return lambda * synapse.weight;
    }
}