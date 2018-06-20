namespace ArtificialIntelligence
{
	public interface IActivationFunction
	{
		double Calculate(double input);
		double CalculateDerivative(double input);
	}
}
