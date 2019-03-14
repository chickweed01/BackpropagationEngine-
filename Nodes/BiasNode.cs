using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationEngine.Nodes
{
    public class BiasNode : AbstractNode
    {
        private double _val;

        public BiasNode(double val = 1.0)
        {
            _val = val;
        }

        public override double Val { get { return _val; } set { _val = value; } }
        public override List<Connector> OutputConnectors { get; set; }
        public override List<Connector> InputConnectors { get; set; }
    }
}
