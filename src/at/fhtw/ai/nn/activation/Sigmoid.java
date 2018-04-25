package at.fhtw.ai.nn.activation;

/**
 * Most common used sigmoid function for activation.<br/>
 * <code>f(x) = 1 / (1 + e^-x)</code><br/><br/>
 * Derivative:<br/>
 * <code>f'(x) = f(x) * (1 - f(x))</code>
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Sigmoid implements ActivationFunction {

    @Override
    public double activate(double x) {
        return 1.0 / (1.0 + Math.pow(Math.E, -x));
    }

    @Override
    public double derivative(double x) {
        return x * (1.0 - x);
    }
}
