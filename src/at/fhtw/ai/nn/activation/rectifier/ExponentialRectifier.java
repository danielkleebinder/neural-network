package at.fhtw.ai.nn.activation.rectifier;

/**
 * Exponential rectifier activation function.<br>
 * <code>f(x) = a*(e^x-1) for x < 0</code><br>
 * <code>f(x) = x for x >= 0</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = f(a,x)+a for x < 0</code><br>
 * <code>f'(x) = 1 for x >= 0</code>
 * <p>
 * Created On: 01.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class ExponentialRectifier extends ParametricRectifier {
    private static final long serialVersionUID = -4255312460170858671L;

    /**
     * Creates a new exponential rectifier with a standard leakiness of 0.01.
     */
    public ExponentialRectifier() {
        this(0.01);
    }

    /**
     * Creates a new parametric rectifier with the given leakiness.
     *
     * @param leakiness Leakiness.
     */
    public ExponentialRectifier(double leakiness) {
        this.leakiness = leakiness;
    }

    @Override
    public double activate(double x) {
        return x < 0 ? (leakiness * (Math.exp(x) - 1)) : x;
    }

    @Override
    public double derivative(double x) {
        return x < 0 ? (x + leakiness) : 1.0;
    }
}