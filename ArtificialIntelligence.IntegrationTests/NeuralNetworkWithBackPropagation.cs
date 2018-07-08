using System;
using ArtificialIntelligence.ActivationFunctions;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Executers;
using ArtificialIntelligence.Initializers;
using ArtificialIntelligence.Learners;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ArtificialIntelligence.IntegrationTests
{
	[TestClass]
	public class NeuralNetworkWithBackPropagation
	{
		private BackPropagationLearner sut;
		private IModelInitializer modelInitializer;
		private Random random;

		[TestInitialize]
		public void TestInitialize()
		{
			var sigmoidActivationFunction = new SigmoidActivationFunction();
			var executer = new FullyConnectedNeuralNetworkExecuter(sigmoidActivationFunction);

			sut = new BackPropagationLearner(executer, sigmoidActivationFunction);
			modelInitializer = new FullyConnectedNeuralNetworkInitializer();
			random = new Random();
		}

		[TestMethod]
		public void LearningForNeuralNetworkWithSigmoidFor10IterationsUsingMinibatchesOfSize100()
		{
			// Arrange
			var iterations = 10;
			var activationFunction = ActivationFunction.Sigmoid;
			var activationCountsPerLayer = new[] { 784, 200, 10 };
			var model = modelInitializer.CreateModel(activationFunction, activationCountsPerLayer, random);
			var trainingInputs = // get mnist training images
			var trainingOutputs = // get mnist training labels

			// Act
			for (var i = 0; i < iterations; i++)
			{
				var miniBatches = // shuffle the training data and split into batches of size 100
				foreach (var batch in miniBatches)
				{
					model = sut.Learn(model, batch);
				}
			}

			// Assert
			// score the trained model against the test data and write the result to the debug window
		}
	}
}
