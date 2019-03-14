using BackpropagationEngine.Nodes;

namespace BackpropagationEngine
{
    public class Connector
    {
        public Connector(double weight = .01, double weightDelta = 0.011)
        {
            Weight = weight;
            WeightDelta = weightDelta;
        }
        public double Weight { get; set; }
        public double WeightDelta { get; set; }
        public AbstractNode FromNode { get; set; }
        public AbstractNode ToNode { get; set; }
    }
}
