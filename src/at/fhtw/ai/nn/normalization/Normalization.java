package at.fhtw.ai.nn.normalization;

import java.io.Serializable;

/**
 * Normalization is used to stretch or squeeze a given data set to a specific range. Neural networks do technically not
 * profit from this method directly. Normalization on input data <b>can</b> increase the learning speed and efficiency.
 * <p>
 * Created On: 29.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public interface Normalization extends Serializable {

    /**
     * Normalizes the given data.
     *
     * @param data Input data.
     * @return Normalized output data.
     */
    double[] normalize(double[] data);
}