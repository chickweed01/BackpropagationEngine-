﻿using System.Collections.Generic;

namespace BackpropagationEngine.Nodes
{
    public class InputNode : AbstractNode
    {
        private double _val;

        public InputNode(double val = 0.0)
        {
            _val = val;
        }
        public override double Val { get { return _val; } set { _val = value; } }
        public override List<Connector> OutputConnectors { get; set; }
        public override List<Connector> InputConnectors { get; set; }
    }
}
