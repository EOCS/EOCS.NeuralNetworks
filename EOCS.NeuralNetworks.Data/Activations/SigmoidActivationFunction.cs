using EOCS.NeuralNetworks.Data.Interfaces;

namespace EOCS.NeuralNetworks.Data.Activations
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        public double Evaluate(double input)
        {
            return 1/(1+Math.Exp(-input));
        }

        public double EvaluateDerivative(double input)
        {
            var temp = Evaluate(input);
            return temp*(1-temp);
        }
    }
}
