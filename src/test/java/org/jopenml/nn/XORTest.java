package org.jopenml.nn;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.jopenml.nn.activationFunctions.ActivationFunction;
import org.jopenml.nn.activationFunctions.Sigmoid;
import org.jopenml.nn.layers.Layer;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

public class XORTest {
	
	private static final double BREAK_ON_ERROR_BELOW = .01;
	private static final int MAX_ITERATIONS = 10_000;
	private static final double MOMENTUM = 0;
	private static final double ETA = .02;
	
	private NeuralNetwork mlp;
	private final Collection<Datum> data;
	private final List<Layer> layers;
	
	public XORTest() {
		data = generateData();
		layers = generateLayers();
	}
	
	@Before
	public void initMLP() {
		mlp = new NeuralNetwork(layers);
	}
	
	@Test
	public void testTrainOnline() {
		int iterations = 0;
		while (mlp.trainOnline(data, ETA, MOMENTUM) > BREAK_ON_ERROR_BELOW) {
			if (++iterations == MAX_ITERATIONS) {
				break;
			}
		}
	}
	
	@Test
	public void testTrainBatch() {
		int iterations = 0;
		while (mlp.trainBatch(data, 1, ETA, MOMENTUM) > BREAK_ON_ERROR_BELOW) {
			if (++iterations == MAX_ITERATIONS) {
				break;
			}
		}
	}
	
	private List<Layer> generateLayers() {
		final List<Layer> layers = new ArrayList<>();
		
		ActivationFunction af = new Sigmoid();
		
		// input layer
		layers.add(new Layer(null, 2, af));
		
		// one hidden Layer
		layers.add(new Layer(layers.get(0), 5, af));
		layers.add(new Layer(layers.get(1), 1, af));
		layers.add(new Layer(layers.get(2), 5, af));
		
		// output layer
		layers.add(new Layer(layers.get(3), 1, af));
		return layers;
	}
	
	private Collection<Datum> generateData() {
		final List<Datum> data = new ArrayList<>();
		
		final double[] target = new double[] { 1 };
		data.add(new Datum(new double[] { 1, 0 }, target));
		data.add(new Datum(new double[] { 0, 1 }, target));
		
		target[0] = 0;
		data.add(new Datum(new double[] { 0, 0 }, target));
		data.add(new Datum(new double[] { 1, 1 }, target));
		
		return data;
	}
	
	@Ignore
	@After
	public void printTrainingSuccess() {
		System.err.println("target 1, value: " + mlp.classify(new double[] { 1, 0 })[0]);
		System.err.println("target 1, value: " + mlp.classify(new double[] { 0, 1 })[0]);
		System.err.println("target 0, value: " + mlp.classify(new double[] { 1, 1 })[0]);
		System.err.println("target 0, value: " + mlp.classify(new double[] { 0, 0 })[0]);
	}
	
}
