package org.jopenml.mlp;
/**
 * Ths Object simply returns the learning rate without any modifications 
  */
public class NoEtaAdjustment implements Eta {

	private double eta;
	/**
	 * Constructor which sets the learning rate.
	 * @param eta the learning rate
	 */
	public NoEtaAdjustment(double eta) {
		this.eta = eta;
	}
	
	/**
	 * Returns the learning rate.
	 * @param lastError This object makes nothing with this variable.
	 * @return The learning rate.
	 */
	public double getEta(double lastError) {
		return eta;
	}
	
	/**
	 * Returns a string-representation of the learning rate.
	 * @return The learning rate as string.
	 */
	@Override
	public String toString(){
		return Double.toString(eta);
	}
}
