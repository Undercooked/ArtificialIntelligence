using System;

namespace ArtificialIntelligence.ActivationFunctions
{
	public class SigmoidActivationFunction : IActivationFunction
	{
		public float Calculate(float input)
		{
			return 1 / (1 + (float)Math.Exp(-input));
		}

		public float CalculateDerivative(float input)
		{
			var sigmoid = Calculate(input);

			return sigmoid * (1 - sigmoid);
		}
	}
}
