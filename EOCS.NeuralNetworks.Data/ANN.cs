using EOCS.NeuralNetworks.Data.Interfaces;

namespace EOCS.NeuralNetworks.Data
{
    public class ANN
    {
        public double[,] HiddenWeights { get; set; }

        public double[] OutputWeights { get; set; }

        public int NumberOfFeatures { get; set; }

        public int NumberOfHiddenUnits { get; set; }

        public IActivationFunction HiddenActivationFunction { get; set; }

        public IActivationFunction OutputActivationFunction { get; set; }

        public ANN(int numberOfFeatures, int numberOfHiddenUnits, IActivationFunction hiddenActivationFunction, IActivationFunction outputActivationFunction) 
        {
            NumberOfFeatures = numberOfFeatures;
            NumberOfHiddenUnits = numberOfHiddenUnits;
            HiddenActivationFunction = hiddenActivationFunction;
            OutputActivationFunction = outputActivationFunction;

            HiddenWeights = new double[NumberOfHiddenUnits, NumberOfFeatures];
            OutputWeights = new double[NumberOfHiddenUnits + 1];
        }
    }
}
