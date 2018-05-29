package at.fhtw.ai.nn.normalization;

/**
 * The min-max normalization is a commonly used algorithm to squeeze all data into a normalized range. All values will
 * be clamped between 0 and 1. The factor by which each value is multiplied to clamp it, is decided by the min and max
 * value of the given data set.
 * <p>
 * Created On: 29.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class MinMax implements Normalization {
    private static final long serialVersionUID = 3420464940340109041L;

    @Override
    public double[] normalize(double[] data) {
        double[] result = new double[data.length];

        // Find min and max values
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < data.length; i++) {
            min = Math.min(min, data[i]);
            max = Math.max(max, data[i]);
        }

        // Normalize data
        for (int i = 0; i < data.length; i++) {
            result[i] = (data[i] - min) / (max - min);
        }
        return result;
    }
}
