using System.Collections.Generic;

namespace BackpropagationEngine.Nodes
{
    public abstract class AbstractNode
    {
        public abstract double Val { get; set; }
        public abstract List<Connector> InputConnectors { get; set; }
        public abstract List<Connector> OutputConnectors { get; set; }
    }
}
