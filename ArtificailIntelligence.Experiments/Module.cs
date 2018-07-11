using Ninject.Modules;

namespace ArtificailIntelligence.Experiments
{
	internal class Module : NinjectModule
	{
		public override void Load()
		{
			Bind<IExperiment>().To<NeuralNetworkWithBackPropagation>();
		}
	}
}
