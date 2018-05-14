package test.genetic.runner;

import at.fhtw.ai.nn.NeuralNetwork;

/**
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Runner extends Entity implements Comparable<Runner> {
    private long lastTunnelTime = 0;
    private long tunnelTime = 500;
    private long tunnelCooldown = 1500;

    public int probability = 15;
    public long score = 0;

    public double left = 0.0;
    public double right = 0.0;
    public boolean tunnelMode = false;
    public boolean dead = false;

    public NeuralNetwork brain;

    public void respawn() {
        lastTunnelTime = 0;
        left = 0.0;
        right = 0.0;
        score = 0;
        tunnelMode = false;
        dead = false;
    }

    public void input(double distanceToNextObstacle, double obstacleSpeed) {
        brain.input(distanceToNextObstacle, obstacleSpeed);
        brain.fireOutput();
    }

    public void update() {
        if (dead) {
            return;
        }

        score++;
        if (tunnelMode) {
            score -= 2;
        }
        if (score < 0) {
            dead = true;
        }
        tunnelMode = tunnel();
    }

    private boolean tunnel() {
        lastTunnelTime = System.currentTimeMillis();
        return isTunnelButtonPressed();
    }

    public boolean isTunnelButtonPressed() {
        return ((int) Math.round(brain.output()[0])) == 1;
    }

    public long getCooldown() {
        return Math.max((lastTunnelTime + tunnelCooldown) - System.currentTimeMillis(), 0);
    }

    @Override
    public int compareTo(Runner o) {
        return (int) (o.score - score);
    }

    @Override
    public String toString() {
        return "Runner{" +
                "score=" + score +
                '}';
    }
}