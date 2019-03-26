using System;

namespace BackpropagationEngine.Activation
{
    public class Sigmoid : IActivationFormulaDelegate
    {
        public double applyActivation(double x)
        {
            if (x < -10.0) return 0.0f;
            else if (x > 10.0) return 1.0f;
            return (1.0 / (1.0 + Math.Exp(-x)));
        }

        public double[] applyActivation(double[] v)
        {
            throw new NotImplementedException();
        }
    }
}
