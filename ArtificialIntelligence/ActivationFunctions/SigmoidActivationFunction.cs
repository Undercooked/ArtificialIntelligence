using System;

namespace ArtificialIntelligence.ActivationFunctions
{
	public class SigmoidActivationFunction : IActivationFunction
	{
		public double Calculate(double input)
		{
			return 1 / (1 + (double)Math.Exp(-input));
		}

		public double CalculateDerivative(double input)
		{
			var sigmoid = Calculate(input);

			return sigmoid * (1 - sigmoid);
		}
	}
}
