using System.Collections.Generic;

namespace BackpropagationEngine.Nodes
{
    public class HiddenNode : AbstractNode
    {
        private double _val;

        public HiddenNode(double val = 0.0)
        {
            _val = val;
        }

        public override double Gradient { get; set; }
        public override double Val { get { return _val; } set { _val = value; } }
        public override List<Connector> OutboundConnectors { get; set; }
        public override List<Connector> InboundConnectors { get; set; }
    }
}
