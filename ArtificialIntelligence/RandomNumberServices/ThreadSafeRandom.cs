using System;

namespace ArtificialIntelligence.RandomNumberServices
{
	public class ThreadSafeRandom
	{
		private static readonly Random seeder = new Random();
		[ThreadStatic]
		private static Random generator;

		private Random Generator
		{
			get
			{
				if (generator == null)
				{
					int seed;
					lock (seeder)
					{
						seed = seeder.Next();
					}
					generator = new Random(seed);
				}

				return generator;
			}
		}

		public int Next()
		{
			return Generator.Next();
		}

		public int Next(int maxValue)
		{
			return Generator.Next(maxValue);
		}

		public int Next(int minValue, int maxValue)
		{
			return Generator.Next(minValue, maxValue);
		}

		public void NextBytes(byte[] buffer)
		{
			Generator.NextBytes(buffer);
		}

		public double NextDouble()
		{
			return Generator.NextDouble();
		}
	}
}
