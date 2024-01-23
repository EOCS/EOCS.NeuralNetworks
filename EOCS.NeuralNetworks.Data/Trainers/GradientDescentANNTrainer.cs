using EOCS.NeuralNetworks.Data.Activations;
using EOCS.NeuralNetworks.Data.Interfaces;

namespace EOCS.NeuralNetworks.Data.Trainers
{
    /// <summary>
    /// DO NOT USE THAT IN PRODUCTION !! ONLY FOR ILLUSTRATION PURPOSES
    /// </summary>
    public class GradientDescentANNTrainer : IANNTrainer
    {
        private ANN _ann; 

        public void Train(DataSet set)
        {
            var numberOfFeatures = set.Features.Count;
            var numberOfHiddenUnits = 20;
            var activationFunction = new SigmoidActivationFunction();

            _ann = new ANN(numberOfFeatures, numberOfHiddenUnits, activationFunction, activationFunction);

            Fit(set);
        }

        public double Predict(DataToPredict record)
        {
            var numberOfHiddenUnits = _ann.NumberOfHiddenUnits + 1;

            var a = new double[numberOfHiddenUnits];
            var z = new double[numberOfHiddenUnits];

            // Forward propagate
            z[0] = 1.0;
            for (var j = 1; j <= _ann.NumberOfHiddenUnits; j++)
            {
                a[j] = 0.0;
                for (var i = 0; i < _ann.NumberOfFeatures; i++)
                {
                    var data = record.Data.ElementAt(i);
                    a[j] = a[j] + _ann.HiddenWeights[j-1, i]*data.Value;
                }
                z[j] = _ann.HiddenActivationFunction.Evaluate(a[j]);
            }

            var b = 0.0;
            for (var j = 0; j < numberOfHiddenUnits; j++)
                b = b + _ann.OutputWeights[j] * z[j];

            var y = _ann.OutputActivationFunction.Evaluate(b);

            return y;
        }

        #region Private Methods

        private void Fit(DataSet set)
        {
            var numberOfHiddenUnitsWithBiases = _ann.NumberOfHiddenUnits + 1;

            var a = new double[numberOfHiddenUnitsWithBiases];
            var z = new double[numberOfHiddenUnitsWithBiases];
            var delta = new double[numberOfHiddenUnitsWithBiases];

            var nu = 0.005;

            // Initialize
            var rnd = new Random();
            for (var i = 0; i < _ann.NumberOfFeatures; i++)
            {
                for (var j = 0; j < _ann.NumberOfHiddenUnits; j++)
                {
                    _ann.HiddenWeights[j, i] = rnd.NextDouble();
                }
            }

            for (var j = 0; j < numberOfHiddenUnitsWithBiases; j++)
                _ann.OutputWeights[j] = rnd.NextDouble();

            for (var n = 0; n < 1000; n++)
            {
                foreach (var record in set.Records)
                {
                    // Forward propagate
                    z[0] = 1.0;
                    for (var j = 1; j <= _ann.NumberOfHiddenUnits; j++)
                    {
                        a[j] = 0.0;
                        for (var i = 0; i < _ann.NumberOfFeatures; i++)
                        {
                            var feature = set.Features[i];
                            a[j] = a[j] + _ann.HiddenWeights[j-1, i]*record.Data[feature];
                        }
                        z[j] = _ann.HiddenActivationFunction.Evaluate(a[j]);
                    }

                    var b = 0.0;
                    for (var j = 0; j < numberOfHiddenUnitsWithBiases; j++)
                        b = b + _ann.OutputWeights[j] * z[j];

                    var y = _ann.OutputActivationFunction.Evaluate(b);

                    // Evaluate the error for the output
                    var d = y - record.Target;

                    // Backpropagate this error
                    for (var j = 0; j < numberOfHiddenUnitsWithBiases; j++)
                        delta[j] = d * _ann.OutputWeights[j] * _ann.HiddenActivationFunction.EvaluateDerivative(a[j]);

                    // Evaluate and utilize the required derivatives
                    for (var j = 0; j < numberOfHiddenUnitsWithBiases; j++)
                    {
                        _ann.OutputWeights[j] = _ann.OutputWeights[j] - nu * d * z[j];
                    }

                    for (var j = 1; j <= _ann.NumberOfHiddenUnits; j++)
                    {
                        for (var i = 0; i < _ann.NumberOfFeatures; i++)
                        {
                            var feature = set.Features[i];
                            _ann.HiddenWeights[j-1, i] = _ann.HiddenWeights[j-1, i] - nu * delta[j]*record.Data[feature];
                        }
                    }
                }
            }
        }

        #endregion
    }
}
