namespace ArtificialIntelligence
{
	public interface IActivationFunction
	{
		public float Calculate(float input);
		public float CalculateDerivative(float input);
	}
}
