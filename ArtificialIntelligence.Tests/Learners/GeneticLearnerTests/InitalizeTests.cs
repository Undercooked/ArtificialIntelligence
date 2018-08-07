using System;
using ArtificialIntelligence.Learners;
using ArtificialIntelligence.Models;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ArtificialIntelligence.Tests.Learners.GeneticLearnerTests
{
	[TestClass]
	public class InitalizeTests
	{
		private GeneticLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			sut = new GeneticLearner();
		}

		[TestMethod]
		public void InitializeCorrectlySetsTheModelPropLearnerIsCorrectlyInitialized()
		{
			// Arrange
			var model = new FullyConnectedNeuralNetworkModel();

			// Act
			sut.Initialize(model);

			// Assert
			sut.Model.Should().BeSameAs(model);
		}
	}
}
