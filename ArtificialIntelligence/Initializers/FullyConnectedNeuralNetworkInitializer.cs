using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Models;
using ArtificialIntelligence.RandomNumberServices;

namespace ArtificialIntelligence.Initializers
{
	public class FullyConnectedNeuralNetworkInitializer : IModelInitializer
	{
		private readonly ThreadSafeRandom random;

		public FullyConnectedNeuralNetworkInitializer(ThreadSafeRandom random)
		{
			this.random = random;
		}

		public FullyConnectedNeuralNetworkModel CreateModel(int[] activationCountsPerLayer, ActivationFunction activationFunction)
		{
			return new FullyConnectedNeuralNetworkModel
			{
				ActivationCountsPerLayer = activationCountsPerLayer,
				ActivationFunction = activationFunction,
				BiasLayers = GenerateInitialBiases(activationCountsPerLayer),
				WeightLayers = GenerateInitialWeights(activationCountsPerLayer)
			};
		}

		public FullyConnectedNeuralNetworkModel CreateModel(int[] activationCountsPerLayer, ActivationFunction activationFunction, double[][] biasLayers, double[][,] weightLayers)
		{
			return new FullyConnectedNeuralNetworkModel
			{
				ActivationCountsPerLayer = activationCountsPerLayer,
				ActivationFunction = activationFunction,
				BiasLayers = biasLayers,
				WeightLayers = weightLayers
			};
		}

		private double[][] GenerateInitialBiases(int[] activationCountsPerLayer)
		{
			var numberOfBiasLayers = activationCountsPerLayer.Length - 1;
			var biases = new double[numberOfBiasLayers][];

			for (var biasLayerIndex = 0; biasLayerIndex < biases.Length; biasLayerIndex++)
			{
				var numberOfBiases = activationCountsPerLayer[biasLayerIndex + 1];
				biases[biasLayerIndex] = new double[numberOfBiases];
				for (var biasIndex = 0; biasIndex < biases[biasLayerIndex].Length; biasIndex++)
				{
					biases[biasLayerIndex][biasIndex] = random.NextDouble() * 2 - 1;
				}
			}

			return biases;
		}

		private double[][,] GenerateInitialWeights(int[] activationCountsPerLayer)
		{
			var numberOfLayersOfWeights = activationCountsPerLayer.Length - 1;
			var weights = new double[numberOfLayersOfWeights][,];

			for (var layerIndex = 0; layerIndex < numberOfLayersOfWeights; layerIndex++)
			{
				var inputNeuronCount = activationCountsPerLayer[layerIndex];
				var outputNeuronCount = activationCountsPerLayer[layerIndex + 1];
				weights[layerIndex] = new double[inputNeuronCount, outputNeuronCount];

				for (var inputNeuronIndex = 0; inputNeuronIndex < inputNeuronCount; inputNeuronIndex++)
				{
					for (var outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++)
					{
						weights[layerIndex][inputNeuronIndex, outputNeuronIndex] = random.NextDouble() * 2 - 1;
					}
				}
			}

			return weights;
		}
	}
}
