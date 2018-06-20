using ArtificialIntelligence.Enums;

namespace ArtificialIntelligence.Models
{
	public class FullyConnectedNeuralNetworkModel
	{
		public double[][] BiasLayers { get; set; }
		public double[][,] WeightLayers { get; set; }
		public ActivationFunction ActivationFunction { get; set; }
	}
}
