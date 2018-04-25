package at.fhtw.ai.nn.activation;

/**
 * Sinusoid activation function.<br/>
 * <code>f(x) = sin(x)</code><br/><br/>
 * Derivative:<br/>
 * <code>f'(x) = cos(x)</code>
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Sinusoid implements ActivationFunction {

    @Override
    public double activate(double x) {
        return Math.sin(x);
    }

    @Override
    public double derivative(double x) {
        return Math.cos(x);
    }
}
