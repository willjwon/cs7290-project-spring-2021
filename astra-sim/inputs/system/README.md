*  **scheduling-policy**: (LIFO/FIFO) 
	* The order we proritize collectives according based on their time of arrival.
        LIFO means that most recently created collectives have higher priority. While
	FIFO is the reverse.
*   **endpoint-delay **: (int)
	* The time NPU spends processing a message after receiving it in terms of cycles.
*   **packet-routing**: (software/hardware)
	* software means that the ubderlying network only can support neighbor-to-neighbor 
	direct communication and software needs to handle non-neighbor communication.
        hardware means that the network protocol allows non-neighbor direct communication.
*  **active-chunks-per-dimension: **: (int)
	* This corresponds to the Maximum number of chunks we like execute in parallel on
	each logical dimesnion of topology.
*  **preferred-dataset-splits**: (int)
	* The number of chunks we divide each collective into.
*  **boost-mode**: (0/1)
	* 0 means all nodes are simulated. 1 means that only the nodes
	directly engaged with node 0 (node 0 collects all of the stats)
	are simulated. It is a technique to reduce the simulation time
	by trading some (minor) accuracy.
* **all-reduce-implementation:**: (Dimension0Collective_Dimension1Collective_...\_DimensionNCollective)
	* Here we can create a multiphase colective all-reduce algorithm and directly specify
	the collective algorithm type for each logical dimension. The available options (algorithms) are:
	ring, direct, doubleBinaryTree, oneRing, oneDirect.
	For example, "ring_doubleBinaryTree" means we create a 
	logical topology with 2 dimensions and we perform ring algorithm
	on the first dimension followed by double binary tree on the second
	dimension for the all-reduce pattern. Hence the number of physical dimension should be
	equal to the number of logical dimensions. The only exceptions are oneRing/oneDirect
	where we assume no matter how many physical dimensions we have, we create a one big logical
	ring/direct(AllToAll) topology where all NPUs are connected and perfrom a one phase ring/direct algorithm.
	Note that oneRing and oneDirect is not available for Garnet Backend in this version. 
* **reduce-scatter-implementation:**: (Dimension0CollectiveAlg_Dimension1CollectiveAlg_...\_DimensionNCollectiveAlg)
	* The same as "all-reduce-implementation:" but for reduce-scatter collective. 
	The available options (algorithms) are: ring, direct, oneRing, oneDirect.
* **reduce-scatter-implementation:**: (Dimension0CollectiveAlg_Dimension1CollectiveAlg_...\_DimensionNCollectiveAlg)
	* The same as "all-reduce-implementation:" but for reduce-scatter collective. 
	The available options are: ring, direct, oneRing, oneDirect.
* **all-gather-implementation:**: (Dimension0CollectiveAlg_Dimension1CollectiveAlg_...\_DimensionNCollectiveAlg)
	* The same as "all-reduce-implementation:" but for all-gather collective. 
	The available options (algorithms) are: ring, direct, oneRing, oneDirect.
* **all-to-all-implementation:**: (Dimension0CollectiveAlg_Dimension1CollectiveAlg_...\_DimensionNCollectiveAlg)
	* The same as "all-reduce-implementation:" but for all-to-all collective. 
	The available options (algorithms) are: ring, direct, oneRing, oneDirect.  
* **collective-optimization**: (baseline/localBWAware)
	* baseline issues allreduce across all dimensions to handle
	allreduce of single chunk. While localBWAware issues a 
	reduce-scatter on  local dimension first (to beak data size)
	followed by allreduce on other dimensions, followed by final allgather
	on the local dimension.
* **direct-collective-window**: (int)
	* This parameter tells NPU how many simultaneous messages it should send in parallel (for each chunk)
	to its peers for direct algorithm (if selected). Each NPU simultaneously sends messages up to this value
	and waits for receives of messages before it further sends more messages. If -1 is set, then it sends to
	all peers at the same time before waiting for any received messages.
	
*NOTE: The default clock cycle period is 1ns (1 Ghz feq). This value is defined inside Sys.hh.
One can change it to any number. It will be a configurable command line parameter in the later
versions.*
