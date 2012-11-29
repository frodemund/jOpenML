package org.jopenml.mlp;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.jopenml.mlp.activationFunctions.Sigmoid;
import org.jopenml.mlp.layers.Layer;
import org.junit.Before;
import org.junit.Test;

public class XORTest {
	
	private MLP mlp;
	private final Collection<Datum> data;
	
	public XORTest() {
		data = generateData();
	}
	
	@Before
	public void initMLP() {
		final List<Layer> layers = new ArrayList<>();
		layers.add(new Layer(null, 2, new Sigmoid()));
		layers.add(new Layer(layers.get(0), 4, new Sigmoid()));
		layers.add(new Layer(layers.get(1), 1, new Sigmoid()));
		
		mlp = new MLP(layers);
	}
	
	@Test
	public void testRunOnline() {
		int iterations = 0;
		while (mlp.runOnline(data, .01, 0) > .0000001) {
			if (++iterations == 1_000_000) {
				break;
			}
		}
		
		testMapping(data, mlp);
	}
	
	@Test
	public void testRunBatch() {
		int iterations = 0;
		while (mlp.runBatch(data, data.size(), .01, 0) > .0000001) {
			if (++iterations == 1_000_000) {
				break;
			}
		}
		
		testMapping(data, mlp);
	}
	
	private void testMapping(Collection<Datum> data, MLP mlp) {
		System.err.println("testing");
		for (final Datum datum : data) {
			final double[] result = mlp.classify(datum.getData());
			System.err.println("Target: " + datum.getTarget()[0] + " | Result: " + result[0]);
		}
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
