package org.jopenml.mlp;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.jopenml.mlp.activationFunctions.Sigmoid;
import org.jopenml.mlp.layers.Layer;
import org.junit.Before;
import org.junit.Test;

public class XORTest {
	
	private static final double BREAK_ON_ERROR_BELOW = .0001;
	private static final int MAX_ITERATIONS = 1_000_000;
	private static final double ETA = .01;
	
	private MLP mlp;
	private final Collection<Datum> data;
	private final List<Layer> layers;
	
	public XORTest() {
		data = generateData();
		layers = generateLayers();
	}
	
	@Before
	public void initMLP() {
		mlp = new MLP(layers);
	}
	
	@Test
	public void testTrainOnline() {
		int iterations = 0;
		while (mlp.trainOnline(data, ETA, 0) > BREAK_ON_ERROR_BELOW) {
			if (++iterations == MAX_ITERATIONS) {
				break;
			}
		}
		
		testMapping(data, mlp);
	}
	
	@Test
	public void testTrainBatch() {
		int iterations = 0;
		while (mlp.trainBatch(data, 1, ETA, 0) > BREAK_ON_ERROR_BELOW) {
			if (++iterations == MAX_ITERATIONS) {
				break;
			}
		}
		
		testMapping(data, mlp);
	}
	
	private void testMapping(Collection<Datum> data, MLP mlp) {
		for (final Datum datum : data) {
			final double[] result = mlp.classify(datum.getData());
			System.err.println("Target: " + datum.getTarget()[0] + " | Result: " + result[0]);
		}
	}
	
	private List<Layer> generateLayers() {
		final List<Layer> layers = new ArrayList<>();
		
		// input layer
		layers.add(new Layer(null, 2, new Sigmoid()));
		
		// one hidden Layer
		layers.add(new Layer(layers.get(0), 4, new Sigmoid()));
		
		// output layer
		layers.add(new Layer(layers.get(1), 1, new Sigmoid()));
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
	
}
