using System;
using ArtificialIntelligence.Genetic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace ArtificialIntelligence.Tests.Genetic.GeneticLearnerTests
{
	[TestClass]
	public class LearnTests
	{
		private const int populationSize = 10;
		private const int selectionSize = 4;
		private Random random;
		private Mock<IModelInitializer> mockModelInitializer;
		private Mock<IModelExecuter> mockModelExecuter;
		private Mock<IModelBreeder> mockModelBreeder;
		private GeneticLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			random = new Random();
			mockModelInitializer = new Mock<IModelInitializer>(MockBehavior.Strict);
			mockModelExecuter = new Mock<IModelExecuter>(MockBehavior.Strict);
			mockModelBreeder = new Mock<IModelBreeder>(MockBehavior.Strict);

			sut = new GeneticLearner(populationSize, selectionSize, random, mockModelInitializer.Object, mockModelExecuter.Object, mockModelBreeder.Object);
		}

		[TestMethod]
		public void SingleIterationOfLearnWithActivationFunctionReturnCorrectResult()
		{
		}
	}
}
