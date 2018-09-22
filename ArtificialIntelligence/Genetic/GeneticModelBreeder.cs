using System.Linq;
using ArtificialIntelligence.Models;
using ArtificialIntelligence.RandomNumberServices;

namespace ArtificialIntelligence.Genetic
{
	public class GeneticModelBreeder : IModelBreeder
	{
		private const double mutationThreshold = 0.1;
		private const double fractionOfChromosomeToMutate = 0.01;

		private readonly IModelInitializer modelInitializer;
		private readonly ThreadSafeRandom random;

		public GeneticModelBreeder(IModelInitializer modelInitializer, ThreadSafeRandom random)
		{
			this.modelInitializer = modelInitializer;
			this.random = random;
		}

		public FullyConnectedNeuralNetworkModel Breed(FullyConnectedNeuralNetworkModel mother, FullyConnectedNeuralNetworkModel father)
		{
			var childBiasLayers = MergeBiasLayers(mother.BiasLayers, father.BiasLayers);
			var childWeightLayers = MergeWeightLayers(mother.WeightLayers, father.WeightLayers);
			var childModel = modelInitializer.CreateModel(mother.ActivationCountsPerLayer, mother.ActivationFunction, childBiasLayers, childWeightLayers);
			var mutationChance = random.NextDouble();

			if (mutationChance < mutationThreshold)
			{
				Mutate(childModel);
			}

			return childModel;
		}

		private double[][] MergeBiasLayers(double[][] motherBiasLayers, double[][] fatherBiasLayers)
		{
			var childBiasLayers = new double[fatherBiasLayers.Length][];

			for (var layerIndex = 0; layerIndex < fatherBiasLayers.Length; layerIndex++)
			{
				childBiasLayers[layerIndex] = new double[fatherBiasLayers[layerIndex].Length];

				for (var biasIndex = 0; biasIndex < fatherBiasLayers[layerIndex].Length; biasIndex++)
				{
					var motherBias = motherBiasLayers[layerIndex][biasIndex];
					var fatherBias = fatherBiasLayers[layerIndex][biasIndex];

					childBiasLayers[layerIndex][biasIndex] = SelectDna(motherBias, fatherBias);
				}
			}

			return childBiasLayers;
		}

		private double[][,] MergeWeightLayers(double[][,] motherWeightLayers, double[][,] fatherWeightLayers)
		{
			var childWeightLayers = new double[motherWeightLayers.Length][,];

			for (var layerIndex = 0; layerIndex < motherWeightLayers.Length; layerIndex++)
			{
				var inputNeuronCount = motherWeightLayers[layerIndex].GetLength(0);
				var outputNeuronCount = motherWeightLayers[layerIndex].GetLength(1);

				childWeightLayers[layerIndex] = new double[inputNeuronCount, outputNeuronCount];

				for (var weightInputIndex = 0; weightInputIndex < inputNeuronCount; weightInputIndex++)
				{
					for (var weightOutputIndex = 0; weightOutputIndex < outputNeuronCount; weightOutputIndex++)
					{
						var motherWeight = motherWeightLayers[layerIndex][weightInputIndex, weightOutputIndex];
						var fatherWeight = fatherWeightLayers[layerIndex][weightInputIndex, weightOutputIndex];

						childWeightLayers[layerIndex][weightInputIndex, weightOutputIndex] = SelectDna(motherWeight, fatherWeight);
					}
				}
			}

			return childWeightLayers;
		}

		private double SelectDna(double motherDna, double fatherDna)
		{
			var geneticDice = random.Next(2);

			return geneticDice == 0 ? motherDna : fatherDna;
		}

		private void Mutate(FullyConnectedNeuralNetworkModel model)
		{
			var biasCount = model.BiasLayers.Sum(biasLayer => biasLayer.Length);
			var weightCount = model.WeightLayers.Sum(weightLayer => weightLayer.Length);
			var totalDnaCount = biasCount + weightCount;
			var dnaToMutateCount = totalDnaCount * fractionOfChromosomeToMutate;

			for (var mutationCount = 0; mutationCount < dnaToMutateCount; mutationCount++)
			{
				if (mutationCount + 1 > dnaToMutateCount && random.NextDouble() > dnaToMutateCount - mutationCount)
				{
					break;
				}

				var biasOrWeight = random.Next(totalDnaCount);

				if (biasOrWeight < biasCount)
				{
					var layerIndex = random.Next(model.BiasLayers.Length);
					var biasIndex = random.Next(model.BiasLayers[layerIndex].Length);

					model.BiasLayers[layerIndex][biasIndex] = random.NextDouble();
				}
				else
				{
					var layerIndex = random.Next(model.WeightLayers.Length);
					var weightInputIndex = random.Next(model.WeightLayers[layerIndex].GetLength(0));
					var weightOutputIndex = random.Next(model.WeightLayers[layerIndex].GetLength(1));

					model.WeightLayers[layerIndex][weightInputIndex, weightOutputIndex] = random.NextDouble();
				}
			}
		}
	}
}
