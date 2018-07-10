using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ArtificialIntelligence.Models;

namespace ArtificialIntelligence.IntegrationTests
{
	internal class MnistDataRepository
	{
		private const string mnistDataFolder = @"..\..\..\MNIST";
		private const string trainingDataPrefix = "training";
		private const string testDataPrefix = "test";

		public IEnumerable<InputOutputPairModel> GetMnistData(bool isTraining)
		{
			var prefix = isTraining ? trainingDataPrefix : testDataPrefix;
			var imageData = ReadImageData($@"{mnistDataFolder}\{prefix}-images.dat");
			var labelData = ReadLabelData($@"{mnistDataFolder}\{prefix}-labels.dat");

			for (var i = 0; i < imageData.Length; i++)
			{
				yield return new InputOutputPairModel
				{
					Inputs = imageData[i].Select(NormalizeImageActivation).ToArray(),
					Outputs = NormalizeLabelActivation(labelData[i])
				};
			}
		}

		private byte[][] ReadImageData(string key)
		{
			var fileBytes = File.ReadAllBytes(key);
			var imageCountBytes = new byte[4];
			var imageHeightBytes = new byte[4];
			var imageWidthBytes = new byte[4];

			Array.Copy(fileBytes, 4, imageCountBytes, 0, 4);
			Array.Copy(fileBytes, 8, imageHeightBytes, 0, 4);
			Array.Copy(fileBytes, 12, imageWidthBytes, 0, 4);
			Array.Reverse(imageCountBytes);
			Array.Reverse(imageHeightBytes);
			Array.Reverse(imageWidthBytes);

			var imageCount = BitConverter.ToInt32(imageCountBytes, 0);
			var imageHeight = BitConverter.ToInt32(imageHeightBytes, 0);
			var imageWidth = BitConverter.ToInt32(imageWidthBytes, 0);
			var images = new byte[imageCount][];

			for (var imageIndex = 0; imageIndex < imageCount; imageIndex++)
			{
				var pixelCountTotal = imageHeight * imageWidth;
				var startIndexInFileBytes = pixelCountTotal * imageIndex + 16;

				images[imageIndex] = new byte[pixelCountTotal];
				Array.Copy(fileBytes, startIndexInFileBytes, images[imageIndex], 0, pixelCountTotal);
			}

			return images;
		}

		private byte[] ReadLabelData(string key)
		{
			var fileBytes = File.ReadAllBytes(key);
			var labelCountBytes = new byte[4];

			Array.Copy(fileBytes, 4, labelCountBytes, 0, 4);
			Array.Reverse(labelCountBytes);

			var labelCount = BitConverter.ToInt32(labelCountBytes, 0);
			var labels = new byte[labelCount];

			Array.Copy(fileBytes, 8, labels, 0, labelCount);

			return labels;
		}

		private double NormalizeImageActivation(byte imageActivation)
		{
			return imageActivation / 255.0;
		}

		private double[] NormalizeLabelActivation(byte label)
		{
			var activations = new double[10];

			activations[label] = 1;

			return activations;
		}
	}
}
