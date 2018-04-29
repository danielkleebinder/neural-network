package at.fhtw.ai.nn.activation;

/**
 * Frequently used hyperbolic tangent (tanh) activation function.<br/>
 * <code>f(x) = tanh(x)</code><br/><br/>
 * Derivative:<br/>
 * <code>f'(x) = 1 - f(x)^2</code>
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class HyperbolicTangent implements ActivationFunction {
    private static final long serialVersionUID = -8549945472858871417L;

    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        return 1.0 - (x * x);
    }
}