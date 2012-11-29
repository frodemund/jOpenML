package org.jopenml.mlp.activationFunctions;

import java.io.Serializable;

import org.jopenml.mlp.Linear;
/**
 * This interface makes the ActivationFunction modular and is implemented by {@link Linear}, {@link Sigmoid}
 * and {@link TangensHyperbolicus}.
 */
public interface ActivationFunction extends Serializable{

	/**
	 * The regular activation function
	 * @param x  The parameter of activation function. 
	 * @return Value of activation function at x
	 */
	public double compute(double x);
	
	/**
	 * The derivative of  the activation function
	 * @param x  The parameter of the derivative of  the activation function
	 * @return Value of the derivative of  the activation function at x
	 */
	public double derivation(double x);

}
