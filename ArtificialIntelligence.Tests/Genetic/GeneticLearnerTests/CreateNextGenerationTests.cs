using System;
using ArtificialIntelligence.ActivationFunctions;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Executers;
using ArtificialIntelligence.Genetic;
using ArtificialIntelligence.Initializers;
using ArtificialIntelligence.RandomNumberServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ArtificialIntelligence.Tests.Genetic.GeneticLearnerTests
{
	[TestClass]
	public class CreateNextGenerationTests
	{
		private const int populationSize = 100;
		private const int selectionSize = 20;
		private static readonly int[] activationCountsPerLayer = new[] { 784, 200, 10 };
		private const ActivationFunction activationFunction = ActivationFunction.Sigmoid;

		private IModelInitializer modelInitializer;
		private IActivationFunction sigmoidActivationFunction;
		private IModelExecuter modelExecuter;
		private IModelBreeder modelBreeder;
		private IPopulationBreeder populationBreeder;
		private ThreadSafeRandom random;
		private GeneticLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			random = new ThreadSafeRandom();
			modelInitializer = new FullyConnectedNeuralNetworkInitializer(random);
			sigmoidActivationFunction = new SigmoidActivationFunction();
			modelExecuter = new FullyConnectedNeuralNetworkExecuter(sigmoidActivationFunction);
			modelBreeder = new GeneticModelBreeder(modelInitializer, random);
			populationBreeder = new PolygamousPopulationBreeder(modelBreeder, random);

			sut = new GeneticLearner(populationSize, selectionSize, modelInitializer, modelExecuter, populationBreeder, random);
		}

		[TestMethod, Ignore]
		public void CallCreateNextGeneration()
		{
			// Arrange
			var model = modelInitializer.CreateModel(activationCountsPerLayer, activationFunction);
			sut.Initialize(model);

			// Act
			// sut.CreateNextGeneration();

			// Assert
		}
	}
}
