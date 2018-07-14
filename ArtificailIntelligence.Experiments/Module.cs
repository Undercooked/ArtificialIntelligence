using System;
using ArtificialIntelligence;
using ArtificialIntelligence.ActivationFunctions;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Executers;
using ArtificialIntelligence.Initializers;
using ArtificialIntelligence.Learners;
using Ninject.Modules;

namespace ArtificailIntelligence.Experiments
{
	internal class Module : NinjectModule
	{
		private readonly Random random = new Random();

		public override void Load()
		{
			Bind<IExecuter>().To<FullyConnectedNeuralNetworkExecuter>();
			Bind<ILearner>().To<BackPropagationLearner>();
			Bind<IDataSource>().To<MnistDataSource>();
			Bind<IModelInitializer>().To<FullyConnectedNeuralNetworkInitializer>();
			Bind<IActivationFunction>().To<SigmoidActivationFunction>().Named(nameof(SigmoidActivationFunction));

			BindExperimentA();
			BindExperimentB();
			BindExperimentC();
		}

		/// <summary>
		/// A fully connected neural network classifier for reading handwritten digits.
		/// 10 iterations of learning, a minibatch size of 100, using sigmoid for the activation function and with 1 hidden layer of size 200.
		/// </summary>
		private void BindExperimentA()
		{
			Bind<IExperiment>()
				.ToConstructor(argSyntax => new NeuralNetworkWithBackPropagationExperiment(
					"Fully connected classifier with minibatch of size 100",
					10,
					100,
					ActivationFunction.Sigmoid,
					new[] { 784, 200, 10 },
					argSyntax.Inject<IExecuter>(),
					argSyntax.Inject<ILearner>(),
					argSyntax.Inject<IModelInitializer>(),
					argSyntax.Inject<IDataSource>(),
					random));
		}

		/// <summary>
		/// A fully connected neural network classifier for reading handwritten digits.
		/// 10 iterations of learning, a minibatch size of 1, using sigmoid for the activation function and with 1 hidden layer of size 200.
		/// </summary>
		private void BindExperimentB()
		{
			Bind<IExperiment>()
				.ToConstructor(argSyntax => new NeuralNetworkWithBackPropagationExperiment(
					"Fully connected classifier with minibatch of size 1",
					10,
					1,
					ActivationFunction.Sigmoid,
					new[] { 784, 200, 10 },
					argSyntax.Inject<IExecuter>(),
					argSyntax.Inject<ILearner>(),
					argSyntax.Inject<IModelInitializer>(),
					argSyntax.Inject<IDataSource>(),
					random));
		}

		/// <summary>
		/// A fully connected neural network classifier for reading handwritten digits.
		/// 10 iterations of learning, a minibatch size of 1, using sigmoid for the activation function and with 1 hidden layer of size 200.
		/// </summary>
		private void BindExperimentC()
		{
			Bind<IExperiment>()
				.ToConstructor(argSyntax => new NeuralNetworkWithBackPropagationExperiment(
					"Fully connected classifier without minibatches",
					10,
					int.MaxValue,
					ActivationFunction.Sigmoid,
					new[] { 784, 200, 10 },
					argSyntax.Inject<IExecuter>(),
					argSyntax.Inject<ILearner>(),
					argSyntax.Inject<IModelInitializer>(),
					argSyntax.Inject<IDataSource>(),
					random));
		}
	}
}
