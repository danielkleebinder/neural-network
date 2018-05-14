package test.loader;

import java.util.function.BiFunction;

/**
 * A simple gray scaled version of the MNIST image.
 * <p>
 * Created On: 03.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class GrayImage extends MnistImage {

    /**
     * Creates a new image.
     *
     * @param label Label.
     * @param data  Data.
     */
    public GrayImage(int label, byte[] data) {
        super(label, data);
    }

    @Override
    protected void convert(byte[] rawData) {
        for (int i = 0; i < rawData.length; i++) {
            int value = rawData[i] & 0xFF;
            value = Math.min(255, value);
            value = Math.max(0, value);
            data[i] = value / 255.0;
        }
    }

    /**
     * A simple factory for creating instances of the image.
     */
    public static class Factory implements BiFunction<Integer, byte[], MnistImage> {
        @Override
        public MnistImage apply(Integer label, byte[] rawData) {
            return new GrayImage(label, rawData);
        }
    }
}
