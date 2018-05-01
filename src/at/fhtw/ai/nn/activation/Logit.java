package at.fhtw.ai.nn.activation;

/**
 * The logit function is mostly used as reverse to the sigmoid.<br>
 * <code>f(x) = ln(x/(1-x))</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = 1/(1+e^-x)</code>
 * <p>
 * Created On: 01.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Logit implements ActivationFunction {
    private static final long serialVersionUID = 2125861323454374205L;

    @Override
    public double activate(double x) {
        return Math.log(x / (1 - x));
    }

    @Override
    public double derivative(double x) {
        return 1.0 / (x - x * x);
    }
}
