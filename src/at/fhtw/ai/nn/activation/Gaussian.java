package at.fhtw.ai.nn.activation;

/**
 * Gaussian activation function.<br/>
 * <code>f(x) = e^-x^2</code><br/><br/>
 * Derivative:<br/>
 * <code>f'(x) = -2*x*e^-x^2</code>
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Gaussian implements ActivationFunction {
    private static final long serialVersionUID = 8752888064634480558L;

    @Override
    public double activate(double x) {
        return Math.exp(-(x * x));
    }

    @Override
    public double derivative(double x) {
        return -2.0 * x * activate(x);
    }
}
