package at.fhtw.ai.nn;

import at.fhtw.ai.nn.utils.Utils;

import java.io.Serializable;

/**
 * A bias is a value which is added to the neurons output before running the activation function.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Bias implements Serializable {
    private static final long serialVersionUID = -3767736239555053482L;

    /**
     * Bias value.
     */
    public double value;

    /**
     * Bias weight.
     */
    public double weight;

    /**
     * Creates a new bias.
     */
    public Bias() {
        this(1.0);
    }

    /**
     * Creates a new bias with the given value.
     *
     * @param value Value.
     */
    public Bias(double value) {
        this(value, Utils.randomWeight());
    }

    /**
     * Creates a new bias with the given value and weight.
     *
     * @param value  Value.
     * @param weight Weight.
     */
    public Bias(double value, double weight) {
        this.value = value;
        this.weight = weight;
    }

    /**
     * Sets the bias value.
     *
     * @param value Value.
     */
    public void setValue(double value) {
        this.value = value;
    }

    /**
     * Returns the bias value.
     *
     * @return Value.
     */
    public double getValue() {
        return value;
    }

    /**
     * Sets the bias weight.
     *
     * @param weight Weight.
     */
    public void setWeight(double weight) {
        this.weight = weight;
    }

    /**
     * Returns the bias weight.
     *
     * @return Weight.
     */
    public double getWeight() {
        return weight;
    }

    /**
     * Computes the bias total value.
     *
     * @return Output.
     */
    public double compute() {
        return value * weight;
    }
}
