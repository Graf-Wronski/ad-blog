---
title: Invert the Powerflow with GNNs
date: 2023-10-12T11:11:04+02:00
author: "Carl Wanninger"
autorAvatar: ""
tags: [gnn, fraunhoferise, machine-learning]
categories: []
image: "/../../img/project-invert-the-porwerflow-with-gnns/thumbnail.png"
---

Topology maps of Distribution Grids are not always 100 % correct. Precise topology knowledge however is requiered for efficient grid expansion and grid enhancement. At Fraunhofer ISE a Graph Neural Network (GNN) based approach is developed in order to verify given grid topologies. This master's project aims at improving the current algorithm.

<!--more-->


<h2 id="introduction">Introduction</h2>

According to the Federal Network Agency, Germanys distribution grids are managed by over 850 distribution system operators <a href="#ref1"> [1] </a>. The methods by which the grid topology maps are maintained differ from paper-based to digital approaches. Due to constant changes in the topology due to hardware decay, switch & breaker operation as well as grid expansion and maintainance works, it is non-trivial to sustain a correct topology map for each low-voltage grid <a href="#ref2"> [2] </a>.
In the past the distribution grid has played a passive role within the energy disribution system: the energy was produced by power plants and supplied in a controlled manner to meet the current energy demands. The transition torwards renewable energy sources, however, redefines it as an active element where power is directly injected, i.e. by balcony power plants, or even stored and managed using the flexibility provided by adjustable loads such as electrical vehicles and heat pumps. The ideal for the modern distribution system is thus the so-called _Smart Grid_ <a href="#ref3"> [3] </a>. In order to control a smart grid with high-precision and in order to be able to expand it cost-efficiently, exact knowledge about the distribution grid's topology is requiered. 
As the grid topology directly influences the flow of electrical power within the grid, one can use the measurement data at hand (Active Power, Reactive Power, Voltage Magnitude and rarely also Voltage Angle) and try and reconstruct the grid's topology. If the grid's loads and topology are given, the task to infer voltage magnitude and angle of the busses can be refered to as _Power-flow_study_ <a href="#ref4"> [4] </a>. Similary, the task to infer the grids topology from DC and load data, could be termed the _Inverse Power-Flow Problem_ (IPFP)  <a href="#ref5"> [5] </a>. Strictly speaking though and given all busses, solving the IPFP would entail constructing the grids _addmitance matrix_ <a href="#ref5"> [6] </a>. This matrix' entries hold the addmitance between each pair of nodes, whereby a 0 indicates that 2 nodes are not connected. At the moment we try to solve the slightly easier task of reconstructing the matrix which has 0's where the addmitance is 0 and 1's everywhere else.
Due to its importance to the renewable energy transition in recent years, several approaches have been undertaken to solve the IPFP. However, from a real-world application's perspective, these approaches usually suffer one of the three following disadvantages:

<ol>
	<li> A non-realistic data-availability is assumed. This means they either assume that every bus is observable <a href="#ref7"> [7] </a> <a href="#ref8"> [8] </a><a href="#ref9"> [9] </a> or that rarely observable magnitudes such as voltage angle are available at most buses. </li>
	<li> Non-electric dimensions such as location and node types are not considered. Omitting this kind of data leads to a bigger search spaces and thus worse scalability to larger grids <a href="#ref10"> [10] </a>. </li>
	<li> Overcomplicating the task by directly trying to infer the grids addmitance matrix. However a grid operator would already hugely benefit from having a grids connection matrix without the specifics of the operated line <a href="#ref11"> [11] </a>. </li>
</ol>

At Fraunhofer ISE we want to develop a method that takes as input a grid topology and bus measurements and decides whether all lines run as indicated. If this is not the case it should mark the errors and suggest a correction. As the input data is naturally graphic and Graph Neural Networks (GNNs) have proven to be a powerful tool in classifying graph data, they appear as canonical contender in solving the IPFP.

<h2 id="method"> Method </h2>

<h3> Classifying Nodes vs. Classifying Edges. </h3>

Coarsly speaking, graph neural networks are neural networks that can either label each node, label each edge or label a whole graph. The problem at hand would appear 
as a archetypical edge classifcation problem. However since we also want to detect missing lines, and do not want to undertake the computational effort of classifying 22.350 possible lines, we decided to design the algorithm as a node classifier. This results in 4 classes. 

<table>
  <tr>
    <th>Class Label</th>
    <th>Description</th>
  </tr>
  <tr>
    <td align="center"> 0 </td>
    <td>This node is not suspicous. The GNN suggests no change to it's lines.</td>
  </tr>
  <tr>
    <td align="center"> 1 </td>
    <td>This node seems to have too many lines attached. Disconnect one.</td>
  </tr>
  <tr>
    <td align="center"> 2 </td>
    <td> This node has the right amount of lines attached, but one partner should be switched.</td>
  </tr>
  <tr>
    <td align="center"> 3 </td>
    <td>This node seems to miss one connection.</td>
  </tr>
</table>

Using these four classes on a grid with potentially one wrongly connected line (see Data) allows unique reconstruction of the grid, should the error be fully discovered.  To illustrate the concept:

<img src="/../../img/project-invert-the-porwerflow-with-gnns/concept.png" alt="figure-2" width="900"/>

An easier formulation of the problem is gained when classes 1 and 3 are merged to one class.

<table>
  <tr>
    <th>Class Label</th>
    <th>Description</th>
  </tr>
  <tr>
    <td align="center"> 0 </td>
    <td>This node is not suspicous. The GNN suggests no change to its lines.</td>
  </tr>
  <tr>
    <td align="center"> 1 </td>
    <td>This node looks suspicous. Something is wrong with its lines.</td>
  </tr>
</table>


<h3 id="gnn"> GNN </h3>

The most general understanding of GNNs, I am aware of, is the _Graph Network Framework_ <a href="#ref12"> [12] </a>. In this masters' project however a special kind of Graph Neural Network was used. While the Master's Thesis might implement the Graph Network Framework, the project and thus the blog entry are focusing on Graph Convolutional Networks with GATv2Conv layers <a href="#ref13"> [13] </a>.

In principal, a GNN can be understood as a series of Graph Transformations, that usually are topology-keeping. To illustrate what 'topology-keeping Graph Transformation' refers to, let us assume we have a network of friends, some of which possess a Spätzle slicer (`owns_slicer`). 

<img src="/../../img/project-invert-the-porwerflow-with-gnns/graph_transform_1.png" alt="figure-2" width="900"/>

Now usually one would always lends one's Spätzle slicer to one's dear friends so if we want to know who in principle has access to a Spätzle slicer, we could do this creating a new graph with a second feature `access_to_slicer` which is one if one has a Spätzle slicer oneself or one has a friend which has a Spätzle slicer. In mathematical terms, with \\(e(x, y) = 1 \\) iff (x,y) is an existing edge and \\(0\\) else and where \((N\)) is the set of all nodes: 

`access_to_slicer`(x) = \\(\min\\)(`owns_slicer(x)` + \\(\Sigma_{y \in N(x)}\\) `owns_slicer`\\((y) * e(x, y) ,   1)) \\)

<img src="/../../img/project-invert-the-porwerflow-with-gnns/graph_transform_2.png" alt="figure-3" width="900"/>

We hereby transformed the graph to a new one, making implicit information explicit while keeping the graph's topology. Now, if you think of for example about using the degree of friendship as an edge weight that models the probability by which your friends will lend you their Spätzle slicer, about lending over two edges or about very thrifty friends, that won't lend you anything, the Graph Transformation becomes very complicated very quickly.
Luckily, doing Machine Learning, we do not want hard-code this transformation anyways but rather learn the best parameters by data. One popular learnable Graph Transformation is above mentioned GATv2Conv. The layer update in GATv2Conv is defined as 

$$ \large{h_{i}^{'} = \sigma\left(\Sigma_{j\in N_i} \alpha_{ij} \cdot Wh_j \right)} $$

Here each new node vector \\(h_i^{'}\\) is gained using an activation function \\(\sigma\\), trainable attention parameters \\(\alpha_{ij}\\) and a trainable weight matrix \\(W\\) that is applied to the old weight vectors of each neighbouring node \\(h_j\\). In out current GNN architecture 3 of these GATv2Conv layers are stacked over each other (which also means that for each classifcation all neighbours with a maximal distance of 3 hops are considered) and used 16 attention heads. These parameters were proposed by my predecessor.

<h2 id="data"> Data </h2>

The dataset contains 8 distribution grids: 5 were synthetically created from OSM-data and 3 were provided by grid operators. Time series data, depicting the power flow on the grids was created, using PyPSA and its Fraunhofer derivative InDiGo <a href="#ref14"> [14] </a>. Those simulations were done for 3 different grid utilization scenarios, depicting the years 2022, 2030 and 2040. In order to simulate errors within the grid's topology, an error generator deletes a line and builds a new one, while leaving the grid ring-free and connected. Labels are given accordingly. 
In order to simulate measurement data on the grid, another power-flow-study is conducted. In the end, for each bus, we gain the difference in voltage angle and voltage magnitude between the power-flow data with and without the built-in error. Also we have the active and reactive load data from the scenario creation earlier available. I will refer to the feature set {voltage angle, voltage magnitude, active load, reactive} as _DC features_.
As busses far away from each other usually are not connected, I further added positional features \\(x\\) & \\(y\\). These are created projecting the busses' longitudes and lattitudes to a local plane and calculating the distance vector to the slack bus <a href="#ref15"> [15] </a>. Furthermore, I added for each bus by hand the attribute bus_type indicating if the bus is within a building or not. This is supposedly helpful information, since the busses of two buildings are typically not connected.


<h2 id="experiments"> Experiments </h2>

<h3 id="impairment"> Impairment </h3>

Since my task within the project was mainly to make the GNN more applicable to real-world data and real-world data usually is both noisy and incomplete, I first wanted to find out how the performance of the existing model would react to these two kinds of _impairments_. I impaired the data with Gaussian Noise (_Noise Impair_), impaired it by deleting certain rows (_Row Impair_) and finally combined those two (_Noise-Row Impair_). I gave the impaired data for evaluation to a four-class-classifier and a binary classifier GNN, which I pretrained on non-impaired data. As you can see in _Figure 2_ the performance dropped for both rather drastically when the GNN was applied to more real-world-like data. 

<img src="/../../img/project-invert-the-porwerflow-with-gnns/q2_binary_vs_non_binary.png" alt="figure-4" width="900"/>

Now, of course, the GNNs did not see impaired data during training. In my second experiment I thus trained 4-class-models on 6 different impairment conditions and compared them to the non-impaired baseline. I evaluated the 7 models without impair aswell as with 5 % and 10 % impairment.

<img src="/../../img/project-invert-the-porwerflow-with-gnns/q3_impairment_during_training.png" alt="figure-5" width="800"/>

Impairing the Data, independent of the method, always made the model more resistant to row impair, not so much more resistant to noise impair. However, if you combine the two impairment methods, than the resulting model is more resistant to both noise and row impair. Also you see the highest performance on non-impaired data is not reached by the baseline, but by impaired models. This indicates, that impairment can also be seen as data augmentation.

In a third experiment, I wanted to know, whether, it helps to differentiate between \\(0\\)'s that mean _no measurement_ and \\(0\\)'s that mean _the value here is \\(0\\)_. I thus extended each tensor by a same-sized tensor, where a \\(1\\) shows that data is original and \\(0\\) shows that the data was impaired. However this had at best now effect (on row impaired data) and at worst dropped the performance significantly (on noise and noise_row impaired data). 

<h3 id="bus_type_and_position"> Bus Type and Position </h3>

Finally, I wanted to know, whether the non-DC, but available features Bus Type aswell as x and y coordinates could improve the performance of the GNN further. I therefore conducted an experiment with all DC features, all DC features + position, all DC features + bus type and all features. 

<img src="/../../img/project-invert-the-porwerflow-with-gnns/q1_effect_on_baseline.png" alt="figure-6" width="500"/>

The positional data seemed to confuse the GNN, while the bus types gave a slight edge over the baseline. If the GNN uses the positional data directly it might be very misleading, since of course the relative position to the slack bus alone is not very helpful. What the GNN would have to learn is to extract the distances, given the relative positions. Since this might be not easy to learn, I restarted the experiment with \\(4500\\) instead of \\(1500\\) epochs.

<h2 id="outlook"> Conclusion & Outlook </h2>

Overall, we are quite confident that GNNs can solve the problem at hand. The method has proven resistant to Data Impairment aswell as further improvement potential. The current best validation score of around 0.67 (reached by a binary classifier on a 25_000 epoch run) is of course far from industrial applicability. One may not forget, however, that the final tasks might be in some regards even simpler than the current setup. In order to reach a even more realistic pipeline and better performance, for my Master's Thesis I have planed inter alia the following improvements:

<h3 id="switches"> Switches </h3>

The current GNN can only process data with lines, but not with switches. However the majoriy of LV-grids contain switches and they are a common source of error in grid topologies. Their implementation will be the next step.

<h3 id="error-generator"> Error Generator </h3>

The algorithm that takes a grid and creates a copy containing an error, is very basic. It only knows one type of error and always adds exactly one error. Real-world data can of course have multiple errors and they do not have to be of the type _one connection is drawn between the wrong busses_ but can be, for example, also of the type _one connection is missing_. A more sophisticated error generator would help to make the GNN more applicable to real-world data.

<h3 id="offline-processing"> Offline Processing </h3>

The typical use-case of the GNN would be a scheduled grid expansion. In this scenario, you want to make sure you fully understand the grids current topology so that it can be improved in the most efficient way. The GNN therefore can work offline and grid-specific. One could leverage this use-case and fine-tune the GNN on different variants of the grid that it is going to evaluate. This should theoretically mean a huge performance boost. 

<h3 id="snapshot-aggregation"> Snapshot Aggregation </h3> 

As of now the GNN evaluates single snapshots. However - as it can be used offline - it would also be possible to evaluate, for example, whole days. The implementation of a clever snapshot-aggregation and the usage of multiple snapshots as input should further boost the GNNs performance.

<h3 id="gnf"> Graph-Network Framework </h3> 

Lastly, I would like to implement the Graph-Network framework. It is more general than the current GATv2Conv stack and allows a more refined use of edge features.


<footer>
    <h2 id="footnote-label"> References </h2>
    <ol>
        <li id="ref1">Bundesnetzagentur, 2022,  Monitoring Report 2022, p. 43; https://www.bundesnetzagentur.de/EN/Areas/Energy/DataCollection_Monitoring/start.html, accesed on 03.11.2023</li>
        <li id="ref2"> Deka et al., 2023, „Learning Distribution Grid Topologies“.</li>
        <li id="ref3"> https://en.wikipedia.org/wiki/Smart_grid </li>
        <li id="ref4"> https://en.wikipedia.org/wiki/Power-flow_study </li>
        <li id="ref5"> Yuan et al., 2016, „Inverse Power Flow Problem“. </li>
        <li id="ref6"> https://en.wikipedia.org/wiki/Nodal_admittance_matrix </li>
        <li id="ref7"> Deka et al., 2016 „Estimating distribution grid topologies“. </li>
        <li id="ref8"> Yu et al., 2018 „PaToPa: A Data-Driven Parameter and Topology Joint Estimation Framework in Distribution Grids“. </li>
        <li id="ref9"> Cavraro et al., 2017 „Voltage Analytics for Power Distribution Network Topology Verification“. </li>
        <li id="ref10"> One can imagine a typical distribution grid with 75 households and, adding around the same amount of underground nodes, around 150 busses. Mathematically speaking there are than 149 * 150 = 22.350 possible lines and thus 2^(22.350) theoretic grid topologies (without line specifics such as addmitance). However, given that distribution grid are typically radialy structured around the transformator, no nodes are disconnected and households are not connected with each other, the set of real topologies is drastically smaller.</li>
        <li id="ref11"> Ardakanian et al., 2017, „On Identification of Distribution Grids“. </li>
        <li id="ref12"> Battaglia et al., 2019, „Relational inductive biases, deep learning, and graph networks“. </li>
        <li id="ref13"> Brody et al., 2022, „How Attentive are Graph Attention Networks?“ </li>
        <li id="ref14"> https://pypsa.org/ </li>
        <li id="ref15"> https://en.wikipedia.org/wiki/Slack_bus </li>
    </ol>
</footer>

