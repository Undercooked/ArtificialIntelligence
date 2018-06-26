using System;
using ArtificialIntelligence.ActivationFunctions;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ArtificialIntelligence.Tests.ActivationFunctions.SigmoidActivationFunctionTests
{
	[TestClass]
	public class CalculateTests
	{
		private SigmoidActivationFunction sut;

		[TestInitialize]
		public void TestInitialize()
		{
			sut = new SigmoidActivationFunction();
		}

		[TestMethod]
		public void CalculateReturnsCorrectResultForZeroInput()
		{
			// Arrange
			var input = 0;

			// Act
			var result = sut.Calculate(input);

			// Assert
			result.Should().Be(0.5);
		}

		[TestMethod]
		public void CalculateReturnsCorrectResultForNegativeInput()
		{
			// Arrange
			var input = -1.23;

			// Act
			var result = sut.Calculate(input);

			// Assert
			result.Should().Be(0.2261814257305462);
		}

		[TestMethod]
		public void CalculateReturnsCorrectResultForPositiveInput()
		{
			// Arrange
			var input = 0.12;

			// Act
			var result = sut.Calculate(input);

			// Assert
			result.Should().Be(0.52996405176457173);
		}
	}
}
