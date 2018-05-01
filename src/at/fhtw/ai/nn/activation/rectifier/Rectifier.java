package at.fhtw.ai.nn.activation.rectifier;

import at.fhtw.ai.nn.activation.ActivationFunction;

/**
 * Simple rectifier activation function.<br>
 * <code>f(x) = max(0,x)</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = 0 for x < 0</code><br>
 * <code>f'(x) = 1 for x >= 0</code>
 * <p>
 * Created On: 01.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Rectifier implements ActivationFunction {
    private static final long serialVersionUID = -5904252180643741013L;

    /**
     * Leakiness for rectifier activation function.
     */
    protected double leakiness = 0.0;

    @Override
    public double activate(double x) {
        return x < 0 ? (leakiness * x) : x;
    }

    @Override
    public double derivative(double x) {
        return x < 0 ? leakiness : 1.0;
    }
}
