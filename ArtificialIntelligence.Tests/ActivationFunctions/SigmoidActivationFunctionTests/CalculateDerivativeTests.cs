using ArtificialIntelligence.ActivationFunctions;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ArtificialIntelligence.Tests.ActivationFunctions.SigmoidActivationFunctionTests
{
	[TestClass]
	public class CalculateDerivativeTests
	{
		private SigmoidActivationFunction sut;

		[TestInitialize]
		public void TestInitialize()
		{
			sut = new SigmoidActivationFunction();
		}

		[TestMethod]
		public void CalculateDerivativeReturnsCorrectResultForZeroInput()
		{
			// Arrange
			var input = 0;

			// Act
			var result = sut.CalculateDerivative(input);

			// Assert
			result.Should().Be(0.25);
		}

		[TestMethod]
		public void CalculateDerivativeReturnsCorrectResultForPositiveInput()
		{
			// Arrange
			var input = 3.21;

			// Act
			var result = sut.CalculateDerivative(input);

			// Assert
			result.Should().Be(0.037286382347523961);
		}

		[TestMethod]
		public void CalculateDerivativeReturnsCorrectResultForNegativeInput()
		{
			// Arrange
			var input = -0.98;

			// Act
			var result = sut.CalculateDerivative(input);

			// Assert
			result.Should().Be(0.19842185807035073);
		}
	}
}
