package at.fhtw.ai.nn.loss;

import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.activation.ActivationFunction;

/**
 * Cross entropy loss function.<br>
 * <code>C = -z * log(v) - (1 - z) * log(1 - v)</code><br>
 * where z is the expected value and v the actual value.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class CrossEntropy implements LossFunction {

    /**
     * Adjustment factor to better approximate loss function.
     */
    private double adjustment = 0.1;

    /**
     * Creates a new cross entropy loss function with an adjustment factor of <code>0.1</code>.
     */
    public CrossEntropy() {
        this(0.1);
    }

    /**
     * Creates a new cross entropy loss function with the given adjustment factor.
     *
     * @param adjustment Adjustment factor.
     */
    public CrossEntropy(double adjustment) {
        this.adjustment = adjustment;
    }

    @Override
    public double compute(Neuron neuron, double expectedValue) {
        double actualValue = neuron.value;
        double t = expectedValue - actualValue;

        ActivationFunction activationFunction = neuron.getActivationFunction();
        if (activationFunction != null && activationFunction.isStochasticDerivative()) {
            return t * adjustment;
        }

        double b = actualValue * (1.0 - actualValue);
        return (t / b) * activationFunction.derivative(neuron);
    }

    /**
     * Sets the adjustment factor. The default value is <code>0.1</code>. An adjustment factor of <code>0.25</code> is a
     * very good approximation for the quadratic loss function using the sigmoid activation.
     *
     * @param adjustment Adjustment factor.
     */
    public void setAdjustment(double adjustment) {
        this.adjustment = adjustment;
    }

    /**
     * Returns the adjustment factor.
     *
     * @return Adjustment factor.
     */
    public double getAdjustment() {
        return adjustment;
    }
}