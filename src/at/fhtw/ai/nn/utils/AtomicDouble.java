package at.fhtw.ai.nn.utils;

import java.util.Objects;

/**
 * Hold a double value for concurrency.
 * <p>
 * Created On: 27.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class AtomicDouble {

    /**
     * Atomic double value.
     */
    public double value;

    /**
     * Creates a new atomic double with initial value 0.
     */
    public AtomicDouble() {
        this(0.0);
    }

    /**
     * Creates a new atomic double with the given initial value.
     *
     * @param value Initial value.
     */
    public AtomicDouble(double value) {
        this.value = value;
    }

    /**
     * Sets the value.
     *
     * @param value Value.
     */
    public void setValue(double value) {
        this.value = value;
    }

    /**
     * Returns the value.
     *
     * @return Value.
     */
    public double getValue() {
        return value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AtomicDouble that = (AtomicDouble) o;
        return Double.compare(that.value, value) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(value);
    }

    @Override
    public String toString() {
        return new StringBuilder()
                .append("AtomicDouble{value=")
                .append(value)
                .append('}')
                .toString();
    }
}