package org.jopenml.mlp;

import java.util.ArrayList;
import java.util.Arrays;
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
		layers.add(new Layer(layers.get(0), 1, new Sigmoid()));
		
		mlp = new MLP(layers);
	}
	
	@Test
	public void testRunOnline() {
		double err;
		while ((err = mlp.runOnline(data, 0.1, 0)) > .001) {
			System.err.println(err);
		}
		testMapping(data, mlp);
	}
	
	@Test
	public void testRunBatch() {
		double err;
		while ((err = mlp.runBatch(data, 4, 0.001, 0)) > .001) {
			System.err.println(err);
		}
		testMapping(data, mlp);
	}
	
	private void testMapping(Collection<Datum> data, MLP mlp) {
		for (final Datum datum : data) {
			final double[] result = mlp.classify(datum.getData());
			if (Arrays.equals(result, datum.getTarget())) {
				System.err.print("Correct - ");
			}
			
			System.err.println(result[0]);
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
