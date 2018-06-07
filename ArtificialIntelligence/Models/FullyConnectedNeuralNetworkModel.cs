using ArtificialIntelligence.Enums;

namespace ArtificialIntelligence.Models
{
	public class FullyConnectedNeuralNetworkModel
	{
		public float[][] BiasLayers { get; set; }
		public float[][,] WeightLayers { get; set; }
		public ActivationFunction ActivationFunction { get; set; }
	}
}
