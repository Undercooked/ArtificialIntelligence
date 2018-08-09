using System;
using System.Linq;
using System.Threading.Tasks;
using ArtificialIntelligence.Models;

namespace ArtificialIntelligence.Learners
{
	public class GeneticLearner : ILearner
	{
		private readonly int populationSize;
		private readonly Random random;
		private readonly IModelInitializer modelInitializer;
		private readonly IExecuter executer;

		private FullyConnectedNeuralNetworkModel[] population;
		private double[] populationCosts;
		private object populationCostLock = new object();

		public FullyConnectedNeuralNetworkModel Model { get; private set; }

		public GeneticLearner(int populationSize, Random random, IModelInitializer modelInitializer, IExecuter executer)
		{
			this.populationSize = populationSize;
			this.random = random;
			this.modelInitializer = modelInitializer;
			this.executer = executer;
			population = new FullyConnectedNeuralNetworkModel[populationSize];
		}

		public void Initialize(FullyConnectedNeuralNetworkModel model)
		{
			Model = model;
			population[0] = model;

			var activationCountsPerLayer = GetActivationCountsPerLayer();

			for (var populationIndex = 1; populationIndex < populationSize; populationIndex++)
			{
				population[populationIndex] = modelInitializer.CreateModel(model.ActivationFunction, activationCountsPerLayer.ToArray(), random);
			}
		}

		public void Learn(InputOutputPairModel[] batch)
		{
			Parallel.ForEach(batch, inputOutputPair =>
			{
				CalculateCostForPopulation(inputOutputPair);

				// TODO: Implement remainder of method
			});
		}

		private int[] GetActivationCountsPerLayer()
		{
			var activationCountsPerLayer = Model.BiasLayers.Select(l => l.Length).ToList();
			activationCountsPerLayer.Insert(0, Model.WeightLayers[0].GetLength(0));

			return activationCountsPerLayer.ToArray();
		}

		private void CalculateCostForPopulation(InputOutputPairModel inputOutputPair)
		{
			populationCosts = new double[populationSize];

			for (var individualIndex = 0; individualIndex < population.Length; individualIndex++)
			{
				populationCosts[individualIndex] = 0;

				var model = population[individualIndex];
				var allActivations = executer.Execute(Model, inputOutputPair.Inputs);
				var cost = CalculateCost(inputOutputPair.Outputs, allActivations.Last());

				lock (populationCostLock)
				{
					populationCosts[individualIndex] += cost;
				}
			}
		}

		private double CalculateCost(double[] expectedOutputs, double[] actualOutputs)
		{
			var totalCost = 0.0;

			for (var j = 0; j < actualOutputs.Length; j++)
			{
				var outputDelta = expectedOutputs[j] - actualOutputs[j];
				var cost = Math.Pow(outputDelta, 2);

				totalCost += cost;
			}

			return totalCost;
		}
	}
}
