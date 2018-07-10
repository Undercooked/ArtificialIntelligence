using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ArtificialIntelligence.ActivationFunctions;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Executers;
using ArtificialIntelligence.Initializers;
using ArtificialIntelligence.Learners;
using ArtificialIntelligence.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ArtificialIntelligence.IntegrationTests
{
	[TestClass]
	public class NeuralNetworkWithBackPropagation
	{
		private FullyConnectedNeuralNetworkExecuter executer;
		private BackPropagationLearner learner;
		private IModelInitializer modelInitializer;
		private Random random;
		private MnistDataRepository mnistDataRepository;

		[TestInitialize]
		public void TestInitialize()
		{
			var sigmoidActivationFunction = new SigmoidActivationFunction();

			executer = new FullyConnectedNeuralNetworkExecuter(sigmoidActivationFunction);
			learner = new BackPropagationLearner(executer, sigmoidActivationFunction);
			modelInitializer = new FullyConnectedNeuralNetworkInitializer();
			random = new Random();
			mnistDataRepository = new MnistDataRepository();
		}

		[TestMethod]
		public void LearningForNeuralNetworkWithSigmoidFor10IterationsUsingMinibatchesOfSize100()
		{
			var iterations = 10;
			var batchSize = 100;
			var activationFunction = ActivationFunction.Sigmoid;
			var activationCountsPerLayer = new[] { 784, 200, 10 };
			var model = modelInitializer.CreateModel(activationFunction, activationCountsPerLayer, random);
			var trainingInputOutputPairs = mnistDataRepository.GetMnistData(true);
			var testInputOutputPairs = mnistDataRepository.GetMnistData(false);

			for (var i = 0; i < iterations; i++)
			{
				var score = ScoreModel(model, testInputOutputPairs);

				Debug.WriteLine($"{nameof(LearningForNeuralNetworkWithSigmoidFor10IterationsUsingMinibatchesOfSize100)} iteration {i}: {score}");

				model = TrainModel(batchSize, model, trainingInputOutputPairs);
			}

			var finalScore = ScoreModel(model, testInputOutputPairs);

			Debug.WriteLine($"{nameof(LearningForNeuralNetworkWithSigmoidFor10IterationsUsingMinibatchesOfSize100)} iteration {iterations}: {finalScore}");
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
