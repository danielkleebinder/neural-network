package test.loader;

import java.util.function.BiFunction;

/**
 * A type of MNIST image which converts all gray scale inputs to simple black and white images using the OTSU
 * threshold algorithm.
 * <p>
 * Created On: 03.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class OTSUImage extends MnistImage {

    /**
     * Creates a new OTSU image.
     *
     * @param label Label.
     * @param data  Data.
     */
    public OTSUImage(int label, byte[] data) {
        super(label, data);
    }

    @Override
    protected void convert(byte[] rawData) {
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] = rawData[i] & 0xFF; //convert to unsigned
        }

        int[] histogram = new int[256];

        for (double datum : data) {
            histogram[(int) datum]++;
        }

        double sum = 0;
        for (int i = 0; i < histogram.length; i++) {
            sum += i * histogram[i];
        }

        double sumB = 0;
        int wB = 0;
        int wF = 0;

        double maxVariance = 0;
        int threshold = 0;

        int i = 0;
        boolean found = false;

        while (i < histogram.length && !found) {
            wB += histogram[i];

            if (wB != 0) {
                wF = data.length - wB;

                if (wF != 0) {
                    sumB += (i * histogram[i]);

                    double mB = sumB / wB;
                    double mF = (sum - sumB) / wF;

                    double varianceBetween = wB * Math.pow((mB - mF), 2);

                    if (varianceBetween > maxVariance) {
                        maxVariance = varianceBetween;
                        threshold = i;
                    }
                } else {
                    found = true;
                }
            }

            i++;
        }

        for (i = 0; i < data.length; i++) {
            data[i] = data[i] <= threshold ? 0 : 1;
        }
    }

    /**
     * A simple factory for creating OTSU images.
     */
    public static class Factory implements BiFunction<Integer, byte[], MnistImage> {
        @Override
        public MnistImage apply(Integer label, byte[] rawData) {
            return new OTSUImage(label, rawData);
        }
    }
}