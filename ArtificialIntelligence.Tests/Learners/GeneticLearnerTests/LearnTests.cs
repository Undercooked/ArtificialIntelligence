using ArtificialIntelligence.Learners;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace ArtificialIntelligence.Tests.Learners.GeneticLearnerTests
{
	[TestClass]
	public class LearnTests
	{
		private Mock<IExecuter> mockExecuter;
		private Mock<IActivationFunction> mockSigmoidActivationFunction;
		private GeneticLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			mockExecuter = new Mock<IExecuter>(MockBehavior.Strict);
			mockSigmoidActivationFunction = new Mock<IActivationFunction>(MockBehavior.Strict);

			sut = new GeneticLearner();
		}

		[TestMethod]
		public void SingleIterationOfLearnWithActivationFunctionReturnCorrectResult()
		{
		}
	}
}
