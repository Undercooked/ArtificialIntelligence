using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ArtificialIntelligence;
using ArtificialIntelligence.ActivationFunctions;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Executers;
using ArtificialIntelligence.Initializers;
using ArtificialIntelligence.Learners;
using ArtificialIntelligence.Models;

namespace ArtificailIntelligence.Experiments
{
	public class NeuralNetworkWithBackPropagation : IExperiment
	{
		private const int iterations = 10;
		private const int batchSize = 100;
		private const ActivationFunction activationFunction = ActivationFunction.Sigmoid;
		private static readonly int[] activationCountsPerLayer = new[] { 784, 200, 10 };

		private FullyConnectedNeuralNetworkExecuter executer;
		private BackPropagationLearner learner;
		private IModelInitializer modelInitializer;
		private Random random;
		private MnistDataSource mnistDataSource;

		public NeuralNetworkWithBackPropagation()
		{
			var sigmoidActivationFunction = new SigmoidActivationFunction();

			executer = new FullyConnectedNeuralNetworkExecuter(sigmoidActivationFunction);
			learner = new BackPropagationLearner(executer, sigmoidActivationFunction);
			modelInitializer = new FullyConnectedNeuralNetworkInitializer();
			random = new Random();
			mnistDataSource = new MnistDataSource();
		}

		public void Run()
		{
			var model = modelInitializer.CreateModel(activationFunction, activationCountsPerLayer, random);
			var trainingInputOutputPairs = mnistDataSource.GetData(DataPurpose.Training);
			var testInputOutputPairs = mnistDataSource.GetData(DataPurpose.Test);

			for (var i = 0; i < iterations; i++)
			{
				var score = ScoreModel(model, testInputOutputPairs);

				Console.WriteLine($"{nameof(NeuralNetworkWithBackPropagation)} iteration {i}: {score}");

				model = TrainModel(batchSize, model, trainingInputOutputPairs);
			}

			var finalScore = ScoreModel(model, testInputOutputPairs);

			Console.WriteLine($"{nameof(NeuralNetworkWithBackPropagation)} iteration {iterations}: {finalScore}");
		}

		private FullyConnectedNeuralNetworkModel TrainModel(int batchSize, FullyConnectedNeuralNetworkModel model, IEnumerable<InputOutputPairModel> trainingInputOutputPairs)
		{
			var miniBatches = CreateMiniBatches(trainingInputOutputPairs, batchSize);

			foreach (var batch in miniBatches)
			{
				model = learner.Learn(model, batch);
			}

			return model;
		}

		private double ScoreModel(FullyConnectedNeuralNetworkModel model, IEnumerable<InputOutputPairModel> testInputOutputPairs)
		{
			var correct = 0;
			var incorrect = 0;

			foreach (var pair in testInputOutputPairs)
			{
				var outputs = executer.Execute(model, pair.Inputs).Last();
				var outputLabel = Array.IndexOf(outputs, outputs.Max());
				var expectedLabel = Array.IndexOf(pair.Outputs, pair.Outputs.Max());

				if (outputLabel == expectedLabel)
				{
					correct++;
				}
				else
				{
					incorrect++;
				}
			}

			return (double)correct / (correct + incorrect);
		}

		private IEnumerable<InputOutputPairModel[]> CreateMiniBatches(IEnumerable<InputOutputPairModel> orderedInputOutputPairs, int batchSize)
		{
			var shuffled = orderedInputOutputPairs.OrderBy(ioPair => random.Next()).ToArray();

			for (var i = 0; i < shuffled.Length; i += batchSize)
			{
				yield return shuffled.Skip(i).Take(batchSize).ToArray();
			}
		}
	}
}
