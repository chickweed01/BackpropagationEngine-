using System.Collections.Generic;

namespace BackpropagationEngine.Nodes
{
    public abstract class AbstractNode
    {
        public abstract double Gradient { get; set; }
        public abstract double Val { get; set; }
        public abstract List<Connector> InboundConnectors { get; set; }
        public abstract List<Connector> OutboundConnectors { get; set; }
    }
}
