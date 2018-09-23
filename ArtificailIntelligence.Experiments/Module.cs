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
			Bind<IActivationFunction>().To<SigmoidActivationFunction>().Named(nameof(SigmoidActivationFunction));
			Bind<IPopulationBreeder>().To<PolygamousPopulationBreeder>().Named(nameof(PolygamousPopulationBreeder));
			Bind<IPopulationBreeder>().To<MonogamousPopulationBreeder>().Named(nameof(MonogamousPopulationBreeder));
			Bind<IModelBreeder>().To<GeneticModelBreeder>();

			Bind<ILearner>().To<BackPropagationLearner>().Named(nameof(BackPropagationLearner));
			
			// The size of the population and selection means that every parent will breed with every other parent exacly once to fully populate the next generation
			Bind<ILearner>().ToConstructor(argSyntax => new GeneticLearner(
				28,
				7,
				argSyntax.Inject<IModelInitializer>(),
				argSyntax.Inject<IModelExecuter>(),
				Kernel.Get<IPopulationBreeder>(nameof(PolygamousPopulationBreeder)),
				random))
				.Named($"Polygamous{nameof(GeneticLearner)}");

			// The size of the population and selection means that every parent will be breed only once to one other random parent to create the next generation
			Bind<ILearner>().ToConstructor(argSyntax => new GeneticLearner(
				63,
				42,
				argSyntax.Inject<IModelInitializer>(),
				argSyntax.Inject<IModelExecuter>(),
				Kernel.Get<IPopulationBreeder>(nameof(MonogamousPopulationBreeder)),
				random))
				.Named($"Monogamous{nameof(GeneticLearner)}");

			//BindExperimentA();
			//BindExperimentB();
			//BindExperimentC();
			BindMonogamousGeneticLearnerExperiment();
			BindPolygamousGeneticLearnerExperiment();
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
		/// 25 iterations of learning using a genetic algorithm, no minibatches, using sigmoid for the activation function, 1 hidden layer of size 200 and using a polygamous method of breeding, meaning a survivor can be bred 1 or more time to produce the next generation.
		/// </summary>
		private void BindPolygamousGeneticLearnerExperiment()
		{
			Bind<IExperiment>()
				.ToConstructor(argSyntax => new Experiment(
					"Genetic neural network with no minibatches",
					750,
					int.MaxValue,
					ActivationFunction.Sigmoid,
					new[] { 784, 200, 10 },
					argSyntax.Inject<IModelExecuter>(),
					Kernel.Get<ILearner>($"Polygamous{nameof(GeneticLearner)}"),
					argSyntax.Inject<IModelInitializer>(),
					argSyntax.Inject<IDataSource>()));
		}

		/// <summary>
		/// A fully connected neural network classifier for reading handwritten digits.
		/// 25 iterations of learning using a genetic algorithm, no minibatches, using sigmoid for the activation function, 1 hidden layer of size 200 and using a monogamous method of breeding, meaning each survivor can be only be bred once produce the next generation.
		/// </summary>
		private void BindMonogamousGeneticLearnerExperiment()
		{
			Bind<IExperiment>()
				.ToConstructor(argSyntax => new Experiment(
					"Genetic neural network with no minibatches",
					750,
					int.MaxValue,
					ActivationFunction.Sigmoid,
					new[] { 784, 200, 10 },
					argSyntax.Inject<IModelExecuter>(),
					Kernel.Get<ILearner>($"Monogamous{nameof(GeneticLearner)}"),
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
					Kernel.Get<ILearner>($"Polygamous{nameof(GeneticLearner)}"),
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
					Kernel.Get<ILearner>($"Polygamous{nameof(GeneticLearner)}"),
					argSyntax.Inject<IModelInitializer>(),
					argSyntax.Inject<IDataSource>()));
		}
	}
}
