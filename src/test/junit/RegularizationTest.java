package test.junit;

import at.fhtw.ai.nn.Synapse;
import at.fhtw.ai.nn.regularization.L1;
import at.fhtw.ai.nn.regularization.L2;
import at.fhtw.ai.nn.regularization.Regularization;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RegularizationTest {
    private static final Double DELTA = 0.00001;

    @Test
    void shouldRegularizeCorrectUsingL1() {
        Regularization regularization = new L1(0.01);
        assertEquals(0.01, regularization.compute(new Synapse(null, 0.8)), DELTA);
        assertEquals(-0.01, regularization.compute(new Synapse(null, -0.8)), DELTA);
    }

    @Test
    void shouldRegularizeCorrectUsingL2() {
        Regularization regularization = new L2(0.01);
        assertEquals(0.008, regularization.compute(new Synapse(null, 0.8)), DELTA);
        assertEquals(-0.008, regularization.compute(new Synapse(null, -0.8)), DELTA);
    }
}