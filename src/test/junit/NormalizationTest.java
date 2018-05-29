package test.junit;

import at.fhtw.ai.nn.normalization.Gaussian;
import at.fhtw.ai.nn.normalization.MinMax;
import at.fhtw.ai.nn.normalization.Normalization;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created On: 29.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class NormalizationTest {
    private static final Double DELTA = 0.00001;
    private static final Random RANDOM = new Random();

    private static final double[] testData1 = new double[]{7.3, -9.4, 0.1, 1.3, 5.5, -3.94, -8.1, 10.0, -1.0, 1.3};
    private static final double[] testData2 = new double[]{30, 36, 52, 42};

    @Test
    void shouldCorrectlyComputeMinMax() {
        Normalization normalization = new MinMax();
        double[] result = normalization.normalize(testData1);
        assertEquals(0.8608247, result[0], DELTA);
        assertEquals(0.0, result[1], DELTA);
        assertEquals(1.0, result[7], DELTA);
    }

    @Test
    void shouldCorrectlyComputeGaussian() {
        Normalization normalization = new Gaussian();
        double[] result = normalization.normalize(testData2);
        assertEquals(-1.230914, result[0], DELTA);
    }

    private double[] randomTestData() {
        double[] result = new double[RANDOM.nextInt(100) + 5];
        for (int i = 0; i < result.length; i++) {
            result[i] = RANDOM.nextGaussian() * ((double) RANDOM.nextInt(100));
        }
        return result;
    }
}
