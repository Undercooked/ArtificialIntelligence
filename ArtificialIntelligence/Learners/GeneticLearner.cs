using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ArtificialIntelligence.Models;

namespace ArtificialIntelligence.Learners
{
	public class GeneticLearner : ILearner
	{
		private const double survivorPecentile = 0.2;

		private readonly int populationSize;
		private readonly Random random;
		private readonly IModelInitializer modelInitializer;
		private readonly IExecuter executer;

		private CostModel<FullyConnectedNeuralNetworkModel>[] population;
		private object populationCostLock = new object();

		public FullyConnectedNeuralNetworkModel Model => population[0].Model;

		public GeneticLearner(int populationSize, Random random, IModelInitializer modelInitializer, IExecuter executer)
		{
			this.populationSize = populationSize;
			this.random = random;
			this.modelInitializer = modelInitializer;
			this.executer = executer;
			population = new CostModel<FullyConnectedNeuralNetworkModel>[populationSize];
		}

		public void Initialize(FullyConnectedNeuralNetworkModel model)
		{
			population[0] = new CostModel<FullyConnectedNeuralNetworkModel>(model);

			var activationCountsPerLayer = GetActivationCountsPerLayer();

			for (var populationIndex = 1; populationIndex < populationSize; populationIndex++)
			{
				var member = modelInitializer.CreateModel(model.ActivationFunction, activationCountsPerLayer.ToArray(), random);
				population[populationIndex] = new CostModel<FullyConnectedNeuralNetworkModel>(member);
			}
		}

		public void Learn(InputOutputPairModel[] batch)
		{
			Parallel.ForEach(batch, inputOutputPair =>
			{
				CalculateCostForPopulation(inputOutputPair);
				SortPopulation();

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
			for (var individualIndex = 0; individualIndex < population.Length; individualIndex++)
			{
				var model = population[individualIndex];
				var allActivations = executer.Execute(Model, inputOutputPair.Inputs);
				var cost = CalculateCost(inputOutputPair.Outputs, allActivations.Last());

				lock (populationCostLock)
				{
					population[individualIndex].Cost += cost;
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

		private void SortPopulation()
		{
			Array.Sort(population, (x, y) => x.Cost.CompareTo(y.Cost));
		}

		private CostModel<FullyConnectedNeuralNetworkModel>[] SelectSurvivors(int iteration)
		{
			var selectionSize = (int)Math.Round(populationSize * survivorPecentile);
			var topSelectionSize = (int)Math.Round(selectionSize * 0.7);
			var randomSelectionSize = selectionSize - topSelectionSize;
			var selection = new CostModel<FullyConnectedNeuralNetworkModel>[selectionSize];
			var uniqueRandomIndexes = CreateUniqueRandomIntegers(topSelectionSize, population.Length, randomSelectionSize);

			for (var populationIndex = 0; populationIndex < topSelectionSize; populationIndex++)
			{
				var model = population[populationIndex].Model;
				selection[populationIndex] = new CostModel<FullyConnectedNeuralNetworkModel>(model);
			}

			for (var populationIndex = topSelectionSize; populationIndex < selection.Length; populationIndex++)
			{
				var selectedMemberIndex = uniqueRandomIndexes[populationIndex - topSelectionSize];
				var model = population[selectedMemberIndex].Model;
				selection[populationIndex] = new CostModel<FullyConnectedNeuralNetworkModel>(model);
			}

			return selection.ToArray();
		}

		private List<int> CreateUniqueRandomIntegers(int minValue, int maxValue, int count)
		{
			var uniqueRandomIntegers = new List<int>();

			while (uniqueRandomIntegers.Count < count)
			{
				var randomInt = random.Next(minValue, maxValue);

				if (!uniqueRandomIntegers.Contains(randomInt))
				{
					uniqueRandomIntegers.Add(randomInt);
				}
			}

			return uniqueRandomIntegers;
		}
	}
}
