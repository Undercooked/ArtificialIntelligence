using System;
using System.Linq;
using ArtificialIntelligence.Models;

namespace ArtificialIntelligence.Learners
{
	public class GeneticLearner : ILearner
	{
		private FullyConnectedNeuralNetworkModel[] population;
		private readonly Random random;
		private readonly IModelInitializer modelInitializer;

		public FullyConnectedNeuralNetworkModel Model { get; private set; }

		public GeneticLearner(int populationSize, Random random, IModelInitializer modelInitializer)
		{
			population = new FullyConnectedNeuralNetworkModel[populationSize];
			this.random = random;
			this.modelInitializer = modelInitializer;
		}

		public void Initialize(FullyConnectedNeuralNetworkModel model)
		{
			Model = model;
			population[0] = model;

			var activationCountsPerLayer = model.WeightLayers.Select(l => l.GetLength(0)).ToList();
			activationCountsPerLayer.Add(model.BiasLayers.Last().Length);

			for (var populationIndex = 1; populationIndex < population.Length; populationIndex++)
			{
				population[populationIndex] = modelInitializer.CreateModel(model.ActivationFunction, activationCountsPerLayer.ToArray(), random);
			}
		}

		public void Learn(InputOutputPairModel[] batch)
		{
			throw new NotImplementedException();
		}
	}
}
