using System;
using ArtificialIntelligence.Learners;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace ArtificialIntelligence.Tests.Learners.GeneticLearnerTests
{
	[TestClass]
	public class LearnTests
	{
		private const int populationSize = 10;
		private Random random;
		private Mock<IModelInitializer> mockModelInitializer;
		private Mock<IExecuter> mockExecuter;
		private GeneticLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			random = new Random();
			mockModelInitializer = new Mock<IModelInitializer>(MockBehavior.Strict);
			mockExecuter = new Mock<IExecuter>(MockBehavior.Strict);

			sut = new GeneticLearner(populationSize, random, mockModelInitializer.Object, mockExecuter.Object);
		}

		[TestMethod]
		public void SingleIterationOfLearnWithActivationFunctionReturnCorrectResult()
		{
		}
	}
}
