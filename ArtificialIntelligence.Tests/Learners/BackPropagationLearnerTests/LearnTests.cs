using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Learners;
using ArtificialIntelligence.Models;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace ArtificialIntelligence.Tests.Learners.BackPropagationLearnerTests
{
	[TestClass]
	public class LearnTests
	{
		private Mock<IExecuter> mockExecuter;
		private Mock<IActivationFunction> mockSigmoidActivationFunction;
		private BackPropagationLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			mockExecuter = new Mock<IExecuter>(MockBehavior.Strict);
			mockSigmoidActivationFunction = new Mock<IActivationFunction>(MockBehavior.Strict);

			sut = new BackPropagationLearner(mockExecuter.Object, mockSigmoidActivationFunction.Object);
		}

		[TestMethod]
		public void SingleIterationOfLearnWithActivationFunctionReturnCorrectResult()
		{
			// Arrange
			var activationFunction = ActivationFunction.Sigmoid;
			var model = CreateFullyConnectedNeuralNetworkModel(activationFunction);
			var inputOutputBatches = CreateBatches();
			var expectedResult = new FullyConnectedNeuralNetworkModel
			{
				BiasLayers = new[]
				{
					new[] { -1.2943006116609224, 0.77069126468580773, 0.41209730715748505 }
				},
				WeightLayers = new[]
				{
					new[,]
					{
						{ -0.80537379400399278, 0.41910423130779234, -0.97124654040321312 },
						{ 0.40789139000539582, -0.89706697984869466, 0.46692747025766013 },
						{ -0.7938561825726973, -0.74708704168834117, -0.0907913437911862 },
						{ 0.536301613044516, 0.1534224913289616, -0.76938811941200691 }
					}
				},
				ActivationFunction = activationFunction
			};

			mockExecuter.Setup(m => m.Execute(It.IsAny<FullyConnectedNeuralNetworkModel>(), It.Is<double[]>(it => it.Equals(inputOutputBatches[0].Inputs))))
				.Returns<FullyConnectedNeuralNetworkModel, double[]>((arg1, arg2) => new double[][] { arg2, new[] { 0.23878872335077425, 0.41971501950211154, 0.66660062117031826 } });
			mockExecuter.Setup(m => m.Execute(It.IsAny<FullyConnectedNeuralNetworkModel>(), It.Is<double[]>(it => it.Equals(inputOutputBatches[1].Inputs))))
				.Returns<FullyConnectedNeuralNetworkModel, double[]>((arg1, arg2) => new double[][] { arg2, new[] { 0.51112549346515468, 0.53561880887592039, 0.61608786703901575 } });
			mockSigmoidActivationFunction.Setup(m => m.CalculateDerivative(It.IsAny<double>()))
				.Returns<double>(arg1 => 0.5 * arg1);

			sut.Initialize(model);

			// Act
			sut.Learn(inputOutputBatches);

			// Assert
			sut.Model.Should().NotBeNull();
			sut.Model.Should().BeEquivalentTo(expectedResult);
		}

		[TestMethod]
		public void SingleIterationOfLearnWithoutActivationFunctionReturnCorrectResult()
		{
			// Arrange
			var activationFunction = ActivationFunction.None;
			var model = CreateFullyConnectedNeuralNetworkModel(activationFunction);
			var inputOutputBatches = CreateBatches();
			var expectedResult = new FullyConnectedNeuralNetworkModel
			{
				BiasLayers = new[]
				{
					new[] { -1.7469237523315382, 0.87181075544508735, 0.035472836914632211 }
				},
				WeightLayers = new[]
				{
					new[,]
					{
						{ -1.0562265202161245, 0.46810951163623871, -1.1486458525020145 },
						{ 0.20928221605096742, -0.841987459370628, 0.2539848744314131 },
						{ -1.2309194153735166, -0.67912216088816624, -0.32232313672256552 },
						{ 0.46326568779018962, 0.19574613013194636, -0.9459582954704272 }
					}
				},
				ActivationFunction = activationFunction
			};

			mockExecuter.Setup(m => m.Execute(It.IsAny<FullyConnectedNeuralNetworkModel>(), It.Is<double[]>(it => it.Equals(inputOutputBatches[0].Inputs))))
				.Returns<FullyConnectedNeuralNetworkModel, double[]>((arg1, arg2) => new double[][] { arg2, new[] { 0.23878872335077425, 0.41971501950211154, 0.66660062117031826 } });
			mockExecuter.Setup(m => m.Execute(It.IsAny<FullyConnectedNeuralNetworkModel>(), It.Is<double[]>(it => it.Equals(inputOutputBatches[1].Inputs))))
				.Returns<FullyConnectedNeuralNetworkModel, double[]>((arg1, arg2) => new double[][] { arg2, new[] { 0.51112549346515468, 0.53561880887592039, 0.61608786703901575 } });

			sut.Initialize(model);

			// Act
			sut.Learn(inputOutputBatches);

			// Assert
			sut.Model.Should().NotBeNull();
			sut.Model.Should().BeEquivalentTo(expectedResult);
		}

		private FullyConnectedNeuralNetworkModel CreateFullyConnectedNeuralNetworkModel(ActivationFunction activationFunction)
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
				ActivationFunction = activationFunction
			};
		}

		private InputOutputPairModel[] CreateBatches()
		{
			return new InputOutputPairModel[]
			{
				new InputOutputPairModel
				{
					Inputs = new[] { 0.56, 0.43, 0.99, 0.14 },
					Outputs = new[] { 1.0, 0.0, 0.0 }
				},
				new InputOutputPairModel
				{
					Inputs = new[] { 0.33, 0.78, 0.02, 0.99 },
					Outputs = new[] { 0.0, 1.0, 0.0 }
				}
			};
		}
	}
}
