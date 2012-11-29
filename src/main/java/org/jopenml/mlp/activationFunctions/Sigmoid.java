package org.jopenml.mlp.activationFunctions;

/**
 * This class implements {@link ActivationFunction} with a sigmoid function. The range is from 0 to 1.
 */
public class Sigmoid
		implements ActivationFunction {
	
	private static final long serialVersionUID = -7520433725747069333L;
	
	/**
	 * Computes the corresponding value of the function to the parameter x
	 * 
	 * @return the function's value with x
	 */
	@Override
	public double compute(double x) {
		return (1D / (1D + Math.exp(-x)));
	}
	
	/**
	 * Computes the corresponding value if the function's derivation to the parameter x
	 * 
	 * @return the function's derivation with value x.
	 */
	@Override
	public double derivation(double x) {
		final double e = Math.exp(x);
		final double e2 = e + 1;
		return e / (e2 * e2);
	}
	
	/**
	 * The function's string presentation.
	 * 
	 * @return the name
	 */
	@Override
	public String toString() {
		return "Sigmoid";
	}
	
}
