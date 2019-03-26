using System;

namespace BackpropagationEngine.Activation
{
    public class HyperTan : IActivationFormulaDelegate
    {
        public double applyActivation(double v)
        {
            if (v < -20.0)
                return -1.0;
            else if (v > 20.0)
                return 1.0;
            else
                return Math.Tanh(v);
        }

        public double[] applyActivation(double[] v)
        {
            throw new NotImplementedException();
        }
    }
}
