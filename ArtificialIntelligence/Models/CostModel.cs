namespace ArtificialIntelligence.Models
{
	public class CostModel<T>
	{
		public double Cost { get; set; }
		public T Model { get; set; }

		public CostModel(T model)
			: this(model, 0)
		{
		}

		public CostModel(T model, double cost)
		{
			Model = model;
			Cost = cost;
		}
	}
}
