package at.fhtw.ai.nn.regularization;

/**
 * Abstract basic implementation of the regularization interface.
 * <p>
 * Created On: 01.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public abstract class AbstractRegularization implements Regularization {
    private static final long serialVersionUID = -6571348814563179576L;

    /**
     * Lambda term for regularization.
     */
    protected double lambda = 1e-5;

    @Override
    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    @Override
    public double getLambda() {
        return lambda;
    }
}
