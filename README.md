# NLSTO
Neural Level-Set Topology Optimization Machinery

To run an example with this code:

Run the file Driver.jl to use the NLSTO method to solve a thermal topology optimization problem. You can compare this to the standard parameterization by changing the option "prior" from "neural" to "pixel".
You can also run either a neural or pixel prior using a simp method by changing the option "problem" to "heat_simp"

To change the TO problem, change the background geometry, spaces, weak forms or objectives in the FEProblem.jl file following the syntax seen in Gridap.jl.