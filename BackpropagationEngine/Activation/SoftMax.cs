using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationEngine.Activation
{
    public class SoftMax: IActivationFormulaDelegate
    {
        public double[] applyActivation(double[] oSums)
        {
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max)
                    max = oSums[i];

            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result; // xi sum to 1.0. 
        }

        public double applyActivation(double v)
        {
            throw new NotImplementedException();
        }
    }
}
