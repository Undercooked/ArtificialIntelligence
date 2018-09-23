using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ArtificialIntelligence.Models;
using ArtificialIntelligence.RandomNumberServices;
using NLog;

namespace ArtificialIntelligence.Genetic
{
	public class GeneticLearner : ILearner
	{
		private static readonly ILogger logger = LogManager.GetCurrentClassLogger();

		private readonly int populationSize;
		private readonly int selectionSize;
		private readonly IModelInitializer modelInitializer;
		private readonly IModelExecuter modelExecuter;
		private readonly IPopulationBreeder populationBreeder;
		private readonly object[] populationCostLocks;
		private readonly ThreadSafeRandom random;

		private CostModel<FullyConnectedNeuralNetworkModel>[] population;
		private bool isFirstLearningIteration = true;

		public FullyConnectedNeuralNetworkModel Model => population[0].Model;

		public GeneticLearner(int populationSize, int selectionSize, IModelInitializer modelInitializer, IModelExecuter modelExecuter, IPopulationBreeder populationBreeder, ThreadSafeRandom random)
		{
			this.populationSize = populationSize;
			this.selectionSize = selectionSize;
			this.modelInitializer = modelInitializer;
			this.modelExecuter = modelExecuter;
			this.populationBreeder = populationBreeder;
			this.random = random;
			populationCostLocks = new object[populationSize].Select(o => new object()).ToArray();
			population = new CostModel<FullyConnectedNeuralNetworkModel>[populationSize];
		}

		public void Initialize(FullyConnectedNeuralNetworkModel model)
		{
			population[0] = new CostModel<FullyConnectedNeuralNetworkModel>(model);

			for (var populationIndex = 1; populationIndex < populationSize; populationIndex++)
			{
				var member = modelInitializer.CreateModel(model.ActivationCountsPerLayer, model.ActivationFunction);
				population[populationIndex] = new CostModel<FullyConnectedNeuralNetworkModel>(member);
			}
		}

		public void Learn(InputOutputPairModel[] batch)
		{
			ReportDiversity();
			CalculatePopulationCostsForBatch(batch);
			SortPopulation();
			ReportCosts();
			CreateNextGeneration();

			isFirstLearningIteration = false;
		}

		private void CalculatePopulationCostsForBatch(IEnumerable<InputOutputPairModel> batch)
		{
			var completed = 0;
			var batchCount = batch.Count();

			using (var progressReporter = new Timer(t =>
			{
				var progress = (double)completed / batchCount * 100;

				Console.SetCursorPosition(0, Console.CursorTop);
				Console.Write($"Population cost calculation progress: {progress.ToString("0.0")}%");
			}, null, 0, 1000))
			{
				Parallel.ForEach(batch, inputOutputPair =>
				{
					CalculatePopulationCostsForInputOutputPair(inputOutputPair);
					Interlocked.Increment(ref completed);
				});
			}

			Console.SetCursorPosition(0, Console.CursorTop);
			Console.WriteLine($"Population cost calculation progress: 100.0%");
		}

		private void CalculatePopulationCostsForInputOutputPair(InputOutputPairModel inputOutputPair)
		{
			var individualIndex = isFirstLearningIteration ? 0 : selectionSize;

			for (; individualIndex < population.Length; individualIndex++)
			{
				var model = population[individualIndex].Model;
				var allActivations = modelExecuter.Execute(model, inputOutputPair.Inputs);
				var cost = CalculateCost(inputOutputPair.Outputs, allActivations.Last());

				lock (populationCostLocks[individualIndex])
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

		private void CreateNextGeneration()
		{
			var parentCostModels = SelectSurvivors();
			var parentModels = parentCostModels.Select(p => p.Model).ToArray();
			var childModels = populationBreeder.CreateNextGeneration(parentModels, populationSize);
			var childCostModels = childModels.Select(c => new CostModel<FullyConnectedNeuralNetworkModel>(c)).ToArray();

			Array.Copy(parentCostModels, population, parentCostModels.Length);
			Array.Copy(childCostModels, 0, population, parentCostModels.Length, childCostModels.Length);
		}

		private CostModel<FullyConnectedNeuralNetworkModel>[] SelectSurvivors()
		{
			var topSelectionSize = (int)Math.Round(selectionSize * 0.7);
			var randomSelectionSize = selectionSize - topSelectionSize;
			var selection = new CostModel<FullyConnectedNeuralNetworkModel>[selectionSize];
			var uniqueRandomIndexes = CreateUniqueRandomIntegers(topSelectionSize, population.Length, randomSelectionSize);

			for (var populationIndex = 0; populationIndex < topSelectionSize; populationIndex++)
			{
				selection[populationIndex] = new CostModel<FullyConnectedNeuralNetworkModel>(population[populationIndex].Model, population[populationIndex].Cost);
			}

			for (var populationIndex = topSelectionSize; populationIndex < selection.Length; populationIndex++)
			{
				var selectedMemberIndex = uniqueRandomIndexes[populationIndex - topSelectionSize];
				selection[populationIndex] = new CostModel<FullyConnectedNeuralNetworkModel>(population[selectedMemberIndex].Model, population[selectedMemberIndex].Cost);
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

		private void ReportCosts()
		{
			var populationCostsString = string.Join(", ", population.Select(p => p.Cost.ToString("0.000")));

			logger.Info($"Population cost: {populationCostsString}");
		}

		private void ReportDiversity()
		{
			var totalDistance = 0.0;
			var totalDistanceLock = new object();

			Parallel.For(0, populationSize, qi =>
			{
				for (var pi = qi + 1; pi < populationSize; pi++)
				{
					var q = new List<double>();
					var p = new List<double>();

					q.AddRange(population[qi].Model.BiasLayers.SelectMany(l => l));
					q.AddRange(population[qi].Model.WeightLayers.SelectMany(l => l.Cast<double>()));
					p.AddRange(population[pi].Model.BiasLayers.SelectMany(l => l));
					p.AddRange(population[pi].Model.WeightLayers.SelectMany(l => l.Cast<double>()));

					var distance = EuclideanDistance(q.ToArray(), p.ToArray());

					lock (totalDistanceLock)
					{
						totalDistance += distance;
					}
				}
			});

			var numberOfDistances = (Math.Pow(populationSize, 2) - populationSize) / 2;
			var averageDistance = totalDistance / numberOfDistances;

			logger.Info($"Average diversity: {averageDistance} | Total diversity: {totalDistance}");
		}

		private double EuclideanDistance(double[] q, double[] p)
		{
			return Math.Sqrt(q.Select((qi, i) => Math.Pow(qi - p[i], 2)).Sum());
		}
	}
}
