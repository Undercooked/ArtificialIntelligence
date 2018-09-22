using System;
using ArtificialIntelligence;
using ArtificialIntelligence.ActivationFunctions;
using ArtificialIntelligence.BackPropagation;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Executers;
using ArtificialIntelligence.Genetic;
using ArtificialIntelligence.Initializers;
using ArtificialIntelligence.RandomNumberServices;
using Ninject;
using Ninject.Modules;

namespace ArtificailIntelligence.Experiments
{
	internal class Module : NinjectModule
	{
		private readonly ThreadSafeRandom random = new ThreadSafeRandom();

		public override void Load()
		{
			Bind<IModelExecuter>().To<FullyConnectedNeuralNetworkExecuter>();
			Bind<IDataSource>().To<MnistDataSource>();
			Bind<IModelInitializer>().To<FullyConnectedNeuralNetworkInitializer>();
			Bind<IModelBreeder>().To<GeneticModelBreeder>();
			Bind<IActivationFunction>().To<SigmoidActivationFunction>().Named(nameof(SigmoidActivationFunction));

			Bind<ILearner>().To<BackPropagationLearner>().Named(nameof(BackPropagationLearner));
			Bind<ILearner>().ToConstructor(argSyntax => new GeneticLearner(
				100,
				20,
				argSyntax.Inject<IModelInitializer>(),
				argSyntax.Inject<IModelExecuter>(),
				argSyntax.Inject<IModelBreeder>(),
				random))
				.Named(nameof(GeneticLearner));

			//BindExperimentA();
			//BindExperimentB();
			//BindExperimentC();
			BindExperimentD();
			//BindExperimentE();
			//BindExperimentF();
		}

		/// <summary>
		/// A fully connected neural network classifier for reading handwritten digits.
		/// 10 iterations of learning using back propagation, a minibatch size of 100, using sigmoid for the activation function and with 1 hidden layer of size 200.
		/// </summary>
		private void BindExperimentA()
		{
			Bind<IExperiment>()
				.ToConstructor(argSyntax => new Experiment(
					"Back propagation neural network with minibatch of size 100",
					10,
					100,
					ActivationFunction.Sigmoid,
					new[] { 784, 200, 10 },
					argSyntax.Inject<IModelExecuter>(),
					Kernel.Get<ILearner>(nameof(BackPropagationLearner)),
					argSyntax.Inject<IModelInitializer>(),
					argSyntax.Inject<IDataSource>()));
		}

		/// <summary>
		/// A fully connected neural network classifier for reading handwritten digits.
		/// 10 iterations of learning using back propagation, a minibatch size of 1, using sigmoid for the activation function and with 1 hidden layer of size 200.
		/// </summary>
		private void BindExperimentB()
		{
			Bind<IExperiment>()
				.ToConstructor(argSyntax => new Experiment(
					"Back propagation neural network with minibatch of size 1",
					10,
					1,
					ActivationFunction.Sigmoid,
					new[] { 784, 200, 10 },
					argSyntax.Inject<IModelExecuter>(),
					Kernel.Get<ILearner>(nameof(BackPropagationLearner)),
					argSyntax.Inject<IModelInitializer>(),
					argSyntax.Inject<IDataSource>()));
		}

		/// <summary>
		/// A fully connected neural network classifier for reading handwritten digits.
		/// 10 iterations of learning using back propagation, no minibatches, using sigmoid for the activation function and with 1 hidden layer of size 200.
		/// </summary>
		private void BindExperimentC()
		{
			Bind<IExperiment>()
				.ToConstructor(argSyntax => new Experiment(
					"Back propagation neural network with no minibatches",
					10,
					int.MaxValue,
					ActivationFunction.Sigmoid,
					new[] { 784, 200, 10 },
					argSyntax.Inject<IModelExecuter>(),
					Kernel.Get<ILearner>(nameof(BackPropagationLearner)),
					argSyntax.Inject<IModelInitializer>(),
					argSyntax.Inject<IDataSource>()));
		}

		/// <summary>
		/// A fully connected neural network classifier for reading handwritten digits.
		/// 10 iterations of learning using a genetic algorithm, no minibatches, using sigmoid for the activation function and with 1 hidden layer of size 200.
		/// </summary>
		private void BindExperimentD()
		{
			Bind<IExperiment>()
				.ToConstructor(argSyntax => new Experiment(
					"Genetic neural network with no minibatches",
					10,
					int.MaxValue,
					ActivationFunction.Sigmoid,
					new[] { 784, 200, 10 },
					argSyntax.Inject<IModelExecuter>(),
					Kernel.Get<ILearner>(nameof(GeneticLearner)),
					argSyntax.Inject<IModelInitializer>(),
					argSyntax.Inject<IDataSource>()));
		}

		/// <summary>
		/// A fully connected neural network classifier for reading handwritten digits.
		/// 10 iterations of learning using a genetic algorithm, a minibatch size of 100, using sigmoid for the activation function and with 1 hidden layer of size 200.
		/// </summary>
		private void BindExperimentE()
		{
			Bind<IExperiment>()
				.ToConstructor(argSyntax => new Experiment(
					"Genetic neural network with minibatch of size 100",
					10,
					100,
					ActivationFunction.Sigmoid,
					new[] { 784, 200, 10 },
					argSyntax.Inject<IModelExecuter>(),
					Kernel.Get<ILearner>(nameof(GeneticLearner)),
					argSyntax.Inject<IModelInitializer>(),
					argSyntax.Inject<IDataSource>()));
		}

		/// <summary>
		/// A fully connected neural network classifier for reading handwritten digits.
		/// 10 iterations of learning using a genetic algorithm, a minibatch size of 1, using sigmoid for the activation function and with 1 hidden layer of size 200.
		/// </summary>
		private void BindExperimentF()
		{
			Bind<IExperiment>()
				.ToConstructor(argSyntax => new Experiment(
					"Genetic neural network with minibatch of size 100",
					10,
					1,
					ActivationFunction.Sigmoid,
					new[] { 784, 200, 10 },
					argSyntax.Inject<IModelExecuter>(),
					Kernel.Get<ILearner>(nameof(GeneticLearner)),
					argSyntax.Inject<IModelInitializer>(),
					argSyntax.Inject<IDataSource>()));
		}
	}
}
