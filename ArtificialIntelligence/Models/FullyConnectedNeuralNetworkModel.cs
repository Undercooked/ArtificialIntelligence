using ArtificialIntelligence.Enums;

namespace ArtificialIntelligence.Models
{
	public class FullyConnectedNeuralNetworkModel
	{
		public int[] ActivationCountsPerLayer { get; set; }
		public ActivationFunction ActivationFunction { get; set; }
		public double[][] BiasLayers { get; set; }
		public double[][,] WeightLayers { get; set; }
	}
}
