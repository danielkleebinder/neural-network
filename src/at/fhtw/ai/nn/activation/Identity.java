package at.fhtw.ai.nn.activation;

/**
 * Simple, standard identity activation function. The input equals the output.<br/>
 * <code>f(x) = x</code><br/><br/>
 * Derivative:<br/>
 * <code>f'(x) = 1</code>
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Identity implements ActivationFunction {

    @Override
    public double activate(double x) {
        return x;
    }

    @Override
    public double derivative(double x) {
        return 1;
    }
}
