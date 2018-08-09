using System;
using System.Linq.Expressions;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Learners;
using ArtificialIntelligence.Models;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace ArtificialIntelligence.Tests.Learners.GeneticLearnerTests
{
	[TestClass]
	public class InitalizeTests
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
		public void InitializeCorrectlySetsTheModel()
		{
			// Arrange
			var model = CreateFullyConnectedNeuralNetworkModel();
			var activationCountsPerLayer = new[] { 4, 3 };
			Expression<Func<IModelInitializer, FullyConnectedNeuralNetworkModel>> createModelFunction = m => m.CreateModel(model.ActivationFunction, activationCountsPerLayer, random);

			mockModelInitializer.Setup(createModelFunction)
				.Returns(new FullyConnectedNeuralNetworkModel());

			// Act
			sut.Initialize(model);

			// Assert
			sut.Model.Should().BeSameAs(model);
			mockModelInitializer.Verify(createModelFunction, Times.Exactly(populationSize - 1));
		}

		private FullyConnectedNeuralNetworkModel CreateFullyConnectedNeuralNetworkModel()
		{
			return new FullyConnectedNeuralNetworkModel
			{
				BiasLayers = new[]
				{
					new[] { -0.84167747099030643, 0.66957177392652811, 0.78872177740033789 }
				},
				WeightLayers = new[]
				{
					new[,]
					{
						{ -0.554521067791861, 0.370098950979346, -0.79384722830441179 },
						{ 0.60650056395982421, -0.95214650032676129, 0.67987006608390721 },
						{ -0.35679294977187781, -0.81505192248851621, 0.14074044914019312 },
						{ 0.60933753829884219, 0.11109885252597684, -0.59281794335358673 }
					}
				},
				ActivationFunction = ActivationFunction.Sigmoid
			};
		}
	}
}
