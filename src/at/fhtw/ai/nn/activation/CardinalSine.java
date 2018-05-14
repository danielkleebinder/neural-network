package at.fhtw.ai.nn.activation;

/**
 * Gaussian activation function.<br>
 * <code>f(x) = e^-x^2</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = -2*x*e^-x^2</code>
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class CardinalSine implements ActivationFunction {

    @Override
    public double activate(double x) {
        if (Double.compare(x, 0.0) == 0) {
            return 1.0;
        }
        return Math.sin(x) / x;
    }

    @Override
    public double derivative(double x) {
        return x;
    }
}
