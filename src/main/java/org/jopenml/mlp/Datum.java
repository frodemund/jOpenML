package org.jopenml.mlp;

import java.util.Arrays;

public class Datum {
	
	private final double[] data;
	private double[] mapping;
	
	public Datum(double[] values) {
		data = Arrays.copyOf(values, values.length);
	}
	
	public Datum(double[] values, double[] mapping) {
		data = Arrays.copyOf(values, values.length);
		this.mapping = Arrays.copyOf(mapping, mapping.length);
	}
	
	public double[] getData() {
		return Arrays.copyOf(data, data.length);
	}
	
	public double[] getTarget() {
		return Arrays.copyOf(mapping, mapping.length);
	}
	
	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("Data: ");
		sb.append(data);
		sb.append("\n mapping: ");
		sb.append(mapping);
		return super.toString();
	}
}
