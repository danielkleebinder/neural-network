package at.fhtw.ai.nn.normalization;

/**
 * The gaussian normalization algorithm is based on the gaussian normal distribution and leads to clamped results
 * between -10.0 and 10.0.
 * <p>
 * Created On: 29.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Gaussian implements Normalization {
    private static final long serialVersionUID = 7205914138953684349L;

    @Override
    public double[] normalize(double[] data) {
        double[] result = new double[data.length];

        // Calculate mean
        double sum = 0.0;
        for (int i = 0; i < data.length; i++) {
            sum += data[i];
        }
        double mean = sum / (double) data.length;

        // Compute standard deviation
        sum = 0.0;
        double delta;
        for (int i = 0; i < data.length; i++) {
            delta = data[i] - mean;
            sum += (delta * delta);
        }
        double standardDeviation = Math.sqrt(sum / (double) data.length);

        // Compute normalization
        for (int i = 0; i < data.length; i++) {
            result[i] = (data[i] - mean) / standardDeviation;
        }
        return result;
    }
}
