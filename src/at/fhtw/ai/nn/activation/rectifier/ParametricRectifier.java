package at.fhtw.ai.nn.activation.rectifier;

/**
 * Parametric rectifier activation function.<br>
 * <code>f(x) = ax for x < 0</code><br>
 * <code>f(x) = x for x >= 0</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = ax for x < 0</code><br>
 * <code>f'(x) = 1 for x >= 0</code>
 * <p>
 * Created On: 01.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class ParametricRectifier extends Rectifier {
    private static final long serialVersionUID = -3325085971491993918L;

    /**
     * Creates a new parametric rectifier with a standard leakiness of 0.01 for leaky rectifed linear unit (Leaky ReLU).
     */
    public ParametricRectifier() {
        this(0.01);
    }

    /**
     * Creates a new parametric rectifier with the given leakiness.
     *
     * @param leakiness Leakiness.
     */
    public ParametricRectifier(double leakiness) {
        this.leakiness = leakiness;
    }

    /**
     * Sets the leakiness of the rectifier. A value of 0.01 will produce a "Leaky ReLU".
     *
     * @param leakiness Leakiness.
     */
    public void setLeakiness(double leakiness) {
        this.leakiness = leakiness;
    }

    /**
     * Returns the leakiness of the rectifier.
     *
     * @return Leakiness.
     */
    public double getLeakiness() {
        return leakiness;
    }
}