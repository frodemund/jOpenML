package org.jopenml.mlp;

/**
 * This exception will be thrown by the {@link MLP} if a incorrect configuration is set.
 */
public class BadConfigException extends Exception {
	private static final long serialVersionUID = 1L;
	private static int identifer = 0;
	
	/**
	 * Identifer for a invalid neuron number
	 */
	public static int INVALID_NEURON_NUMBER = 1;
	
	/**
	 * Identifer for a invalid activation function
	 */
	public static int INVALID_ACTIVATION_FUNCTION = 2;

	/**
	 * Identifer for a invalid input dimension
	 */
	public static int INVALID_INPUT_DIMENSION = 3;

	/**
	 * Identifer for a invalid input layer dimension
	 */
	public static int INVALID_INPUT_LAYER_DIMENSION = 4;

	/**
	 * Identifer for a invalid output layer dimension
	 */
	public static int INVALID_OUTPUT_LAYER_DIMENSION = 5;

	/**
	 * Identifer for a invalid eta
	 */
	public static int INVALID_ETA = 6;

	/**
	 * Identifer for a invalid number of values for the bias
	 */
	public static int INVALID_NUMBER_OF_BIAS = 7;
	
	/**
	 * Identifer for a invalid number of activation functions
	 */
	public static int INVALID_NUMBER_OF_FUNCTIONS = 8;
	
	/**
	 * Identifer for a invalid target vector
	 */
	public static int INVALID_TARGET_VECTOR = 9;
	
	/**
	 * Identifer for a invalid error vector
	 */
	public static int INVALID_ERROR_VECTOR = 10;
	
	/**
	 * Identifer for a problem caused by a missing dvv.
	 */
	public static int DVV_NOT_LOADED = 11;
	
	/**
	 * Constructor which only sets the error message.
	 * @param message The error message.
	 */
	public BadConfigException(String message) {
		super(message);
	}
	
	/**
	 * This constructor sets the error message and a identifer describing the message.
	 * @param message The error message.
	 * @param id The corresponding error identifer.
	 */
	public BadConfigException(String message, int id) {
		super(message);
		identifer = id;
	}
	
	/**
	 * Returns the exceptions identifer.
	 * @return The identifer corresponding to the error message.
	 */
	public int getId(){
		return identifer;
	}
	
}
