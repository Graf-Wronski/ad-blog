---
title: Invert the Powerflow with GNNs
date: 2023-10-12T11:11:04+02:00
author: "Carl Wanninger"
autorAvatar: ""
tags: []
categories: []
image: "/concept.PNG"
---

Topology maps of Distribution Grids are not always 100 % correct. Precise topology knowledge however is requiered for efficient grid expansion and grid enhancement. At Fraunhofer ISE a Graph Neural Network (GNN) based approach is developed in order to verify given grid topologies. This master's project aims at improving the current algorithm.

<!--more-->


<h1 id="introduction">Introduction</h1>
According to the Federal Network Agency, Germanys Distribution Grids are managed by over 850 Distribution Grid Operators [SOURCE1]. The methods by which the grid topology maps are maintained differ from paper-based to digital approaches [SOURCE2]. Due to constant changes in the topology due to hardware decay, switch & breaker operation as well as grid expansion and maintainance works, it is non-trivial to sustain a correct topology map for each low-voltage grid [SOURCE3].
In the past the low-voltage (LV) grid has played a passive role within the energy disribution system: the energy was produced by power plants and supplied in a controlled manner to meet the current energy demands. The transition torwards renewable energy sources, however, redefines it as an active element where power is directly injected, i.e. by balcony power plants, or even stored and managed using the load and capacity flexibility provided by for example  electrical vehicles and heat pumps (# SmartGrid) [SOURCE4]. In order to control the LVgrid in a precise manner and in order to be able to expand it cost-efficiently, exact knowledge about the distribution grid's topology is needed.
As the grid topology directly influences the flow of electrical power [SOURCE5] within the grid, one can use the measurement data at hand (Active Power, Reactive Power, sometimes Voltage Magnitude and, if you are really lucky, Voltage Angle) and try and reconstruct the grids topology. The underlying task can be termed the "Inverse Power-Flow Problem" [SOURCE6]. 
Due to its importance to the renewable energy transition in recent years, several approaches have been undertaken to solve the IPFP. However, from a real-world application's perspective, these approaches usually  suffer one of the three following disadvantages:
	
	1. A non-realistic data-availability is assumed. This means they either assume that every bus is observable [SOURCES7-9]or that rarely observable magnitudes such as _voltage angle_ are available at most buses. [SOURCES10-12]
	2. Non-electric dimensions such as location and node types are not considered. Omitting this kind of data leads to a bigger search spaces and thus worse scalability to larger grids.
	3. Overcomplicating the task by directly trying to infer the grids addmitance matrix [EXPLAIN]. However a grid operator would already hugely benefit from having a grids connection matrix without the specifics of the operated line.


At Fraunhofer ISE we want to develop a method that takes as input a grid topology and bus measurements and decides whether all lines run as indicated. If this is not the case it should mark the errors and suggest a correction. As the input data is naturally graphic and Graph Neural Networks (GNNs) have proven to be a powerful tool in classifying graph data, they appear as canonical contender in solving the IPFP.

<h1 id="method"> Method </h1>
<h2> Classifying Nodes vs. Classifying Edges. </h2>
Coarsly speaking, graph neural networks are neural networks that can either label each node, label each edge or label a whole graph. The problem at hand would appear 
as archetypical edge classifcation problem. However since we also want to detect missing lines, and do not want to undertake the computational effort of classifying 22.350 possible lines (*), we decided to design the algorithm as a node classifier. This results in 4 classes. 

    --------- | ----------
    Class 0:  | This node is not suspicous. The GNN suggests no change to it's lines.
    Class 1:  | This node seems to have too many lines attached. Disconect one.
    Class 2:  | This node has the right amount of lines attached, but one partner should be switched.
    Class 3:  | This node seems to miss one connection.

Using these four classes on a grid with potentially one wrongly connected line (see Data) allows unique reconstruction of the grid, should the error be fully discovered. Figure 1 ![Figure 1](/concept.PNG) illustrates the concept. An easier formulation of the problem is gained when classes 1 and 3 are merged to one class.

    --------- | ----------
    Class 0:  | This node is not suspicous. The GNN suggests no change to it's lines.
    Class 1:  | This node seems suspicous. Something is wrong with it's lines.

<h2 id="gnn"> GNN

The most general understanding of GNNs, I am aware of, is the _Graph Network Framework_ [SOURCE 14]. In this masters' project however a special kind of Graph Neural Network was used. While the Master's Thesis might implement the Graph Network Framework, the project and the blog entry are focusing on Graph Convolutional Networks with GATv2Conv layers [SOURCE].

In principal a GNN can be understood as a series of Graph Transformations, that usually are topology-keeping. To illustrate what 'topology-keeping Graph Transformation' refers to, let us assume we have a network of friends some of which possess a "Spätzlehobel". Now usually one would always lends one's "Spätzlehobel" to one's dear friends so if we want to know who in principle has access to a "Spätzlehobel", we could do this creating a new graph with a second feature _has_access_to_a_"Spätzlehobel" which is one if one has a "Spätzlehobel" oneself or one has a friend which has a "Spätzlehobel".  In mathematical terms: AS(x) = min(S(x) + Sum(Neighbours of x: y) S(y) * e(x, y) , 1) where e(x, y) = 1 iff (x,y) in E and 0 else. We now have transformed the graph to a new one, making implicit information explicit while keeping its topology. Now if you think of for example about using the degree of friendship as an edge weight that models the probability by which your friends will lend you their "Spätzlehobel", about lending over two edges or about very thrifty friends, that won't lend you anything, the Graph Transformation becomes very complicated very quickly.
Luckily, doing Machine Learning, we do not want hard-code this transformation anyways but rather learn the best parameters by data. Now, one popular learnable Graph Transformation is GATv2Conv [SOURCE]. The layer update in GATv2Conv is defined as 

---
$$ h_{i}^{'} = \sigma\left\Sigma_{j\in N_i} \alpha_{ij} \cdot Wh_j \right $$.
---

Here each node vector $$h_i$$ is updated using an activation function $$\sigma$$, trainable attention parameters $$\alpha_{ij}$$ and a trainable weight matrix $W$ that is applied to the weight vectors of each neighbour of node $i$. If you are interested in the details of GATv2Conv and want to learn about the advantages of _dynamic attention_ over _static attention_ you should checkout this paper. In my project I stacked 3 of these GATv2Conv layers each others (which also means that for each classifcation all neighbours with a maximal distance of 3 hops are considered) and used 16 attention heads. These parameters were proposed by my predecessor. [SOURCE]

<h1 id="data"> Data </h1>
The dataset contains 8 distribution grids: 5  were synthetically created from OSM-data and 3 were provided by grid operators. Time series data, depicting the power flow on the grids was created, using PyPSA and its derivative InDiGo [Sources]. Those simulations were done for 3 different grid utilization scenarios, depicting the years 2022, 2030 and 2040. In order to simulate errors within the grid's topology, an error generator deletes a line and builds a new one, while leaving the grid ring-free and connected. Labels are given accordingly. 
In order to simulate measurement data on the grid, another power-flow-study is conducted. In the end, for each bus, we gain the difference in voltage angle and voltage magnitude between the power-flow data with and without the built-in error. Also we have the active and reactive load data from the scenario creation earlier available.
As busses far away from each other usually are not connected, I further added positional features x, y. These are created projecting the busses longitudes and lattitudes to a local plane and calculating the distance vector to the slack bus [EXPLAIN]. Furthermore I added for each bus by hand the attribute "bus_type" indicating if the bus is within a building or not. This is supposedly helpful information, since the busses of two buildings are typically not connected.


<h1 id="experiments"> Experiments </h1>

<h2 id="impairment"> Impairment </h2>

Since my task within the project was mainly to make the GNN more applicable to real-world data and real-world data usually is both noisy and incomplete, I first wanted first to find out how the performance of the existing model would react to these two kinds of _impairments_. I impaired the data with Gaussian Noise (_Noise Impair_), impaired it by deleting certain rows (_Row Impair_) and finally combined those two (_Noise-Row Impair_). I gave the impaired data for evaluation to a four-class-classifier and a binary classifier GNN, which I pretrained on non-impaired data. As you can see in _Figure 2_ the performance dropped for both rather drastically when the GNN was applied to more real-world-like data. 

Figure 2 ![Figure 2](/q2_binary_vs_non_binary.png)

Now, of course, the GNNs did not see impaired data during training. In my second experiment I thus trained the models on 6 different impairment conditions and compared them to the non-impaired baseline. I evaluated the 7 models without impair aswell as with 5 % and 10 % impairment.

Figure 3 ![Figure 3](/q3_impairment_during_training.png)

As you can see in Figure 3, impairing the Data, independent of the method, always made the model more resistant to row impair, not so much more resistant to noise impair. However, if you combine the two impairment methods, than the resulting model is more resistant to both noise and row impair. Also you see the highest performance on non-impaired data is not reached by the baseline, but by impaired models. This indicates, that impairment can also be seen as data augmentation.

In a forth experiment, I wanted to know, whether, it helps to differentiate between 0's that mean _no measurement_ and 0's that mean _the value here is 0_. I thus extended each tensor by a same-sized tensor, where a 1 shows that data is original and 0 shows that the data was impaired. However this had at best now effect (on row impaired data) and at worst dropped the performance significantly (on noise and noise_row impaired data). 

<h2 id="bus_type_and_position"> Bus Type and Position </h2>
Finally, I wanted to know, whether the non-electrical, but available features _Bus Type_ aswell as _x_ and _y_ coordinates could improve the performance of the GNN further. I therefore conducted an experiment with all electrical features, all electrical features + position, all electrical features + bus type and all features. The results are as following:


<h1 id="outlook"> Outlook </h1>

As a wrap up I want to give an outlook of the things that need further improvement and will be dealt with in my master thesis.
<h2 id="error-generator"> Error Generator </h2>
The algorithm that takes a grid and creates a copy containing an error, is very basic. It only knows one type of error and always adds exactly one error. Real-world data can of course have multiple errors and they do not have to be of the type 'one connection is drawn between the wrong busses' but can be for example also of the type 'one connection is missing'. A more sophisticated error generator would help to make the GNN more applicable to real-world data.
<h2 id="offline-processing"> Offline Processing </h2>
The typical use-case of the GNN would be an scheduled grid expansion, where you want to make sure you fully understand the grids current topology so that you can improve it in the most efficient way. The GNN therefore can work offline and grid-specific. One could leverage this use-case and fine-tune the GNN on different variants of the grid that it is going to evaluate. This should theoretically mean a huge performance boost. 
<h2 id="snapshot-aggregation"> Snapshot Aggregation </h2> 
As of now the GNN evaluates single snapshots. However - as it can be used offline - it would also be possible to evaluate, for example, whole days. The implementation of a clever snapshot-aggregation and the usage of multiple snapshots as input should further boost the GNNs performance.
<h2 id="switches"> Switches </h2>
The current GNN can only process data with lines, but not with switches. However the majoriy of LV-grids contain switches [SOURCE?] and they are a common source of error in grid topologies. Their implementation will be the next step.
<h2 id="gnf"> Graph-Network Framework </h2> 
Lastly, I would like to implement the Graph-Network framework. It is more general than the current GATv2Conv stack and allows a more refined use of edge features.





Is it possible to verify or even rectify a given grid topolog


Due to the importance with regards to energy transition, several approaches in solving the

Traditional methods, i.e. algorithms that try to reconstruct the electrical properties of the grid do not seem to pay tribute to the fact that not every theoretically possible grid is a real option. 

The connections points within a grid are called "busses" and vary very strongly depending on the size of the supplied locality. But one can imagine a typical LV-grid with 75 households and, adding around the same amount of underground nodes, around 150 busses. Mathematically speaking there are than 149 * 150 = 22.350 possible lines and thus 2^(22.350) theoretic grid topologies. However, given that LV-grids are typically radialy structured around the transformator, no nodes are disconnected and households are not connected with each other, the set of real topologies is drastically smaller.

[SOURCE1]:
[SOURCE2]:
[SOURCE3]:
[SOURCE4]:
[SOURCE5]: https://en.wikipedia.org/wiki/Power-flow_study
[SOURCE6]: Yuan et al. 2016
[SOURCE14]: Battaglia et al 2019