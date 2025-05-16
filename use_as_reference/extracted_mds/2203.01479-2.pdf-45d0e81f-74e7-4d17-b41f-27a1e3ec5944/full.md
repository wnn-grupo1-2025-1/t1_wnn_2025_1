# Weightless Neural Networks for Efficient Edge Inference  

Zachary Susskind∗, Aman Arora†   
Department of Electrical and Computer Engineering   
The University of Texas at Austin   
Igor Dantas Dos Santos Miranda‡   
Department of Electrical and Computer Engineering   
Federal University of Recˆoncavo da Bahia   
Luis Armando Quintanilla Villon§, Rafael Fontella Katopodis   
Department of Electrical and Computer Engineering   
Federal University of Rio de Janeiro   
Leandro Santiago de Ara´ujo   
Department of Electrical and Computer Engineering   
Universidade Federal Fluminense   
Diego Leonel Cadette Dutra, Priscila Machado Vieira Lima, Felipe Maia Galva˜o Franc¸a   
Department of Electrical and Computer Engineering   
Federal University of Rio de Janeiro   
Mauricio Breternitz Jr.   
ISTAR   
ISCTE Instituto Universitario de Lisboa   
Lizy K. John   
Department of Electrical and Computer Engineering   
The University of Texas at Austin  

Abstract—Weightless Neural Networks (WNNs) are a class of machine learning model which use table lookups to perform inference. This is in contrast with Deep Neural Networks (DNNs), which use multiply-accumulate operations. State-of-the-art WNN architectures have a fraction of the implementation cost of DNNs, but still lag behind them on accuracy for common image recognition tasks. Additionally, many existing WNN architectures suffer from high memory requirements. In this paper, we propose a novel WNN architecture, BTHOWeN, with key algorithmic and architectural improvements over prior work, namely counting Bloom filters, hardware-friendly hashing, and Gaussian-based nonlinear thermometer encodings to improve model accuracy and reduce area and energy consumption. BTHOWeN targets the large and growing edge computing sector by providing superior latency and energy efficiency to comparable quantized DNNs. Compared to state-of-the-art WNNs across nine classification datasets, BTHOWeN on average reduces error by more than than $40 \%$ and model size by more than $50 \%$ . We then demonstrate the viability of the BTHOWeN architecture by presenting an FPGAbased accelerator, and compare its latency and resource usage against similarly accurate quantized DNN accelerators, including Multi-Layer Perceptron (MLP) and convolutional models. The proposed BTHOWeN models consume almost $80 \%$ less energy than the MLP models, with nearly $85 \%$ reduction in latency. In our quest for efficient ML on the edge, WNNs are clearly deserving of additional attention.  

Index Terms—weightless neural networks, WNN, WiSARD, neural networks  

# I. INTRODUCTION  

In the last decade, Deep Neural Networks (DNNs) have driven revolutionary improvements in the accuracy of tasks such as image recognition, image classification, speech recognition, and natural language processing. In fact, it is widely acknowledged that modern DNNs can achieve superhuman accuracy on image recognition and classification tasks [1]. However, the implementation of these models is expensive in both memory and computation. Table I shows the number of weights and multiply-accumulate operations (MACs) needed for some widely-known networks. These networks have excellent accuracy, but performing inference with them requires significant memory capacity and MAC computation, which in turn consumes a substantial amount of energy. This may be acceptable on large servers, but in the emerging domain of edge computing, models must be run on small, powerconstrained devices. The amount of weight memory and the number of computations required by these DNNs make them impractical to implement in edge solutions. Consequently, DNNs for edge inference must typically trade off accuracy for reduced complexity through techniques such as pruning and low-precision quantization [2].  

Weightless Neural Networks (WNNs) are an entirely distinct class of neural model, inspired by the decode processing of input signals received by the dendritic trees of biological neurons [3]. WNNs are composed of artificial neurons with discrete (usually binary) inputs and outputs which do not use weights to determine their responses. Instead, WNN neurons, also known as RAM nodes, use Lookup Tables (LUTs) to represent Boolean functions of their inputs as truth tables. Rather than performing arithmetic operations with their inputs, RAM nodes simply concatenate them to form an address, then perform a table lookup to determine their response. A RAM node with $n$ inputs can represent any of the $2 ^ { 2 ^ { n } }$ possible logical functions of its inputs using $2 ^ { n }$ bits of storage.  

TABLE I WEIGHTS AND MACS FOR POPULAR DNNS [4] [5]   


<html><body><table><tr><td>Metric</td><td>LeNet-5</td><td>AlexNet</td><td>VGG-16</td><td>Resnet-50</td><td>OpenPose</td></tr><tr><td>#Weights</td><td>60k</td><td>61M</td><td>138M</td><td>25.5M</td><td>46M</td></tr><tr><td>#MACs</td><td>341k</td><td>724M</td><td>15.5G</td><td>3.9G</td><td>180G</td></tr><tr><td>Year</td><td>1998</td><td>2012</td><td>2014</td><td>2015</td><td>2018</td></tr></table></body></html>  

Foundational research in WNNs occurred from the 1950s through the 1970s. However, WiSARD (Wilkie, Stonham, and Aleksander’s Recognition Device) [6], introduced in 1981 and sold commercially from 1984, was the first WNN to be broadly viable. WiSARD was a pattern recognition machine, specialized for image recognition tasks. Two factors led to its success. First, then-recent advancements in integrated circuit manufacturing allowed for the fabrication of complex devices with large RAMs. Additionally, WiSARD incorporated algorithmic improvements which greatly increased its memory efficiency over simpler WNNs, allowing for the implementation of more sophisticated models. As recent results have formally shown, the VC dimension1 of WiSARD is very large [8], meaning it has a large theoretical capacity to learn patterns. Many subsequent WNNs [9], including the model proposed in this paper, draw inspiration from WiSARD’s basic architecture.  

Training a WNN entails learning logical functions in its component RAM nodes. Both supervised [10] and unsupervised [11] learning techniques have been explored for this purpose. Many training techniques for WNNs directly set values in the RAM nodes. The mutual independence between nodes when LUT entries are changed means that each input in the training set only needs to be presented to the network once. By contrast, most DNN training techniques involve iteratively adjusting weights, and many epochs of training may be needed before a model converges. By leveraging oneshot training techniques, WNNs can be trained up to four orders of magnitude faster than DNNs and other well-known computational intelligence models such as SVM [12].  

Algorithmic and hardware improvements, combined with widespread research efforts, drove rapid and substantial increases in DNN accuracies during the 2010s. The ability to rapidly train large networks on powerful GPUs and the availability of big data fueled an AI revolution which is still taking place. While DNNs drove this revolution, we believe that WNNs are now a concept worth revisiting due to increasing interest in low-power edge inference. WNNs also have potential as tools to accompany DNNs. For instance, it has been demonstrated that WNNs can be used to dramatically speed up the convergence of DNNs during training [13]. There are also many applications where a small network is run first for approximate detection; then, if needed, a larger network is used for more precision [14]. The approximate networks by design do not need high accuracy; high speed and low energy usage are more important considerations. WNNs are perfect for these applications. However, in order to realize the benefits of this class of neural network, work needs to be done to design optimized WNNs with high accuracy and low area and energy costs.  

Microcontroller-based approaches to edge inference, such as tinyML, have attracted a great deal of interest recently due to their ability to use inexpensive off-the-shelf hardware [15]. However, these approaches to machine learning are thousands of times slower than dedicated accelerators.  

In this paper, we explore techniques to improve the accuracy and reduce the hardware requirements of WNNs. These techniques include hardware-efficient counting Bloom filters, hardware implementation of recent algorithmic improvements such as bleaching [16] [17], and a novel nonlinear thermometer encoding. We combine these techniques to create a software model and hardware architecture for WNNs which we call BTHOWeN (Bleached Thermometer-encoded Hashedinput Optimized Weightless Neural Network; pronounced as Beethoven). We present FPGA implementations of inference accelerators for this architecture, discuss their associated tradeoffs, and compare them against prior work in WNNs and against DNNs with similar accuracy.  

Our specific contributions in this paper are as follows:  

1) BTHOWeN, a weightless neural network architecture designed for edge inference, which incorporates novel, hardware-efficient counting Bloom filters, nonlinear thermometer encoding, and bleaching.   
2) Comparison of BTHOWeN with state-of-the-art WiSARD-based WNNs across nine datasets, with a mean $41 \%$ reduction in error and $5 1 \%$ reduction in model size.   
3) An FPGA implementation of the BTHOWeN architecture, which we compare against MLP and CNN models of similar accuracy on the same nine datasets, finding a mean $7 9 \%$ reduction in energy and $84 \%$ reduction in latency versus MLP models. Compared to CNNs of similar accuracy, the energy reduction is over $9 8 \%$ and latency reduction is over $9 9 \%$ .   
4) A toolchain for generating BTHOWeN models, including automated hyperparameter sweeping and bleaching value selection. A second toolchain for converting trained BTHOWeN models to RTL for our accelerator architecture. These are available at: URL omitted for double-blinding.  

The remainder of our paper is organized as followed: In Section II, we provide additional background on WNNs, WiSARD, and prior algorithmic improvements. In Section III, we present the BTHOWeN architecture in detail. In Section IV, we discuss software and hardware implementation details. In Section V, we compare our model architecture against prior memory-efficient WNNs, and compare our accelerator architecture against a prior WNN accelerator and against MLPs and CNNs of comparable accuracy. Lastly, in Section VI, we discuss future work and conclude.  

# II. BACKGROUND AND PRIOR WORK  

# A. Weightless Neural Networks  

Weightless neural networks (WNNs) are a type of neural model which use table lookups for computation. WNNs are sometimes considered a type of Binary Neural Network (BNNs), but their method of operation differs significantly from other BNNs. Most BNNs are based around popcounts, i.e. counting the number of 1s in some bit vector. For instance, the McCulloch-Pitts neuron [18], one of the oldest and simplest neural models, performs a popcount on its inputs and compares the result against a fixed threshold in order to determine its output. More modern approaches first take the XNOR of the input with a learned weight vector, allowing an input to be negated before the popcount occurs [19].  

The fundamental unit of computation in WNNs is the RAM node, an $n$ -input, $2 ^ { n }$ -output lookup table with learned 1-bit entries. Conventionally, all entries in the RAM nodes are initialized to 0. During training, inputs are binarized or discretized using some encoding scheme and then presented to the RAM nodes. The input bits to a node are concatenated to form an address, and the corresponding entry in the node’s LUT is set to 1. Note that presenting the same input to the node again has no effect, since the corresponding bit position has already been set. Therefore, an advantage of this approach is that each training sample only needs to be presented once.  

Lookup tables are able to implement any Boolean function of their inputs. Therefore, in theory, a WNN can be constructed with a single RAM node which takes all (encoded) input features as inputs. However, this approach has two major issues. First, the size of a RAM node grows exponentially with its number of inputs. Suppose we take a dataset such as MNIST [20] and apply a simple encoding strategy such that each of the original inputs is represented using 1 bit. Since images in the MNIST dataset are $2 8 \mathrm { x } 2 8$ , our input vector has 784 bits, and therefore the RAM node requires $2 ^ { 7 8 4 }$ bits of storage, about $1 . 7 * 1 0 ^ { 1 5 6 }$ times the number of atoms in the visible universe. The second issue is that RAM nodes have no ability to generalize: if a single bit is flipped in an input pattern, the node can not recognize it as being similar to a pattern it has seen before.  

A great deal of WNN literature revolves around finding solutions to these two issues. A discussion of many of these approaches can be found in [3] [9] . Unfortunately, many of these techniques require random behavior (e.g. replacing the entries in RAM nodes with Bernoulli random variables), which is challenging to implement in hardware. The WiSARD model addresses both issues with the single-RAM-node model while avoiding the pitfalls of other solutions.  

There are some structural similarities between WNNs and architectural predictors in microprocessors. For instance, using a concatenated input vector to index into a RAM node is conceptually similar to using a branch history register in a table-based branch predictor.  

# B. WiSARD  

WiSARD [6], depicted in Figure 1, is perhaps the most broadly successful weightless neural model. WiSARD is intended primarily for classification tasks, and has a submodel known as a discriminator for each output class. Each discriminator is in turn composed of $n$ -input RAM nodes; for an $I$ -input model, there are $N \equiv I / n$ nodes per discriminator. Inputs are assigned to these RAM nodes using a pseudorandom mapping; typically, as in Figure 1, the same mapping is shared between all discriminators.  

During training, inputs are presented only to the discriminator corresponding to the correct output class, and its component RAM nodes are updated. During inference, inputs are presented to all discriminators. Each discriminator then forms a bit vector from the outputs of its component RAM nodes and performs a popcount on this vector to produce a response value. The index of the discriminator with the highest response is taken to be the predicted class. Figure 2 shows a simplified view of a WiSARD model performing inference. The response from Discriminator 1 is the highest since the input image contains the digit “1”.  

If an input seen during inference is identical to one seen during training, then all RAM nodes of the corresponding discriminator will yield a 1, resulting in the maximum possible response. On the other hand, if a pattern is similar but not identical, then some subset of the RAM nodes may produce a 0, but many will still yield a 1. As long as the response of the correct discriminator is still stronger than the responses of all other discriminators, the network will output a correct prediction. In practice, WiSARD has a far greater ability to generalize than simpler WNN models.  

WiSARD’s performance is directly related to the choice of $n$ . Small values of $n$ give the model a great deal of ability to generalize, while larger values produce more specialized behavior. On the other hand, larger values of $n$ increase the complexity of the Boolean functions that the model can represent [6].  

# C. Bloom Filters  

Although the WiSARD model avoids the state explosion problem inherent in large, simple WNNs, practical considerations still limit the sizes of the individual RAM nodes. Increasing the number of inputs to each RAM node will, up to a point, improve the accuracy of the model; however, the model size will also increase exponentially. Fortunately, the contents of these large RAM nodes are highly sparse, as few distinct patterns are seen during in training relative to the large number of entries available. Prior work has shown that using hashing to map large input sets to smaller RAMs can greatly decrease model size at a minimal impact to accuracy [21].  

![](images/b2b1b954dc3f07df3c77d959d235ce9f5545b3d43721351bbbf45ea74533e6cb.jpg)  
Fig. 1. A depiction of the WiSARD WNN model with $I$ inputs, $M$ classes, and $\scriptstyle n$ inputs per RAM node. $I / n$ RAM nodes are needed per discriminator, for a total of $M ( I / n )$ nodes and $M ( I / n ) 2 ^ { n }$ bits of state.  

![](images/7196df84ef5ccf39672df3fa7d3086e44902707cb203ea8352ee6f680c25b5b8.jpg)  
Fig. 2. A WiSARD model recognizing digits. In this example, the input digit is 1, and the corresponding discriminator produces the strongest response.  

A Bloom Filter [22] is a hash-based data structure for approximate set membership. When presented with an input, a Bloom filter can return one of two responses: 0, indicating that the input is definitely not a member of the set, or 1, indicating that the element is possibly a member of the set. False negatives do not occur, but false positives can occur with a probability that increases with the number of elements in the set and decreases with the size of the underlying data structure [23]. Bloom filters have found widespread application for membership queries in areas such as networking, databases, web caching, and architectural predictions [24]. A recent model, Bloom WiSARD [21], demonstrated that replacing the RAM nodes in WiSARD with Bloom filters improves memory efficiency and model robustness [25].  

Internally, a Bloom Filter is composed of $k$ distinct hash functions, each of which takes an $n$ -bit input and produces an $m$ -bit output, and a $2 ^ { m }$ -bit RAM. When a new value is added to the set represented by the filter, it is passed through all $k$ hash functions, and the corresponding bit positions in the RAM are set. When the filter is checked to see if a value is in the set, the value is hashed, and the filter reports the value as present only if all $k$ of the corresponding bit positions are set.  

# D. Bleaching  

Traditional RAM nodes activate when presented with any pattern they saw during training, even if that pattern was only seen once. This can result in overfitting, particularly for large datasets, a phenomenon known as saturation. Bleaching [17] is a technique which prevents saturation by choosing a threshold $b$ such that nodes only respond to patterns they saw at least $b$ times during training. During training, this requires replacing the single-bit values in the RAM nodes with counters which track how many times a pattern was encountered. After training is complete, a bleaching threshold $b$ can be selected to maximize the accuracy of the network2. Once $b$ has been selected, counter values greater than or equal to $b$ can be statically replaced with 1, and counter values less than $b$ with 0. Therefore, while additional memory is required during training, inference with a bleached WNN introduces no additional overhead.  

In practice, bleaching can substantially improve the accuracy of WNNs. There have been several strategies proposed for finding the optimal bleaching threshold $b$ ; we use a binary search strategy based on the method proposed in [17]. Our approach performs a search between 1 and the largest counter value seen in any RAM node. Thus, both the space and time overheads of bleaching are worst-case logarithmic in the size of the training dataset.  

# E. Thermometer Encoding  

Traditionally, WNNs represent their inputs as 1-bit values, where an input is 1 if it rises above some pre-determined threshold3 and 0 otherwise. However, it is frequently advantageous to use more sophisticated encodings, where each parameter is represented using multiple bits [26]. Integer encodings are not a good choice for WiSARD, since individual bits carry dramatically different amounts of information. In an 8-bit encoding, the most significant bit would carry a great deal of information about the value of a parameter, while the least significant bit would essentially be noise. Since the assignment of bits to RAM nodes is randomized, this would result in some inputs to some RAM nodes being useless.  

In a thermometer encoding, a value is compared against a series of increasing thresholds, with the $i$ ’th bit of the encoded value representing the result of the comparison against the $i$ ’th threshold. Clearly if a value is greater than the $i ^ { \because }$ th threshold, it is also greater than thresholds $\{ 0 \ldots ( i - 1 ) \}$ ; as Figure 3 shows, the encoding resembles mercury passing the markings on an analog thermometer, with bits becoming set from least to most significant as the value increases.  

![](images/49deab3dd04b15c17a76811af560e24bc32f5c81be25875a3c7662c4ba7679a9.jpg)  
Fig. 3. Like the mercury passing the gradations in a thermometer, in a thermometer encoding, bits are set to 1 from least to most significant as the encoded value increases.  

# III. PROPOSED DESIGN: BTHOWEN  

In this paper, we present BTHOWeN, a WNN architecture which improves on the prior work by incorporating (i) counting Bloom filters to reduce model size while enabling bleaching, (ii) an inexpensive hash function which does not require arithmetic operations, and (iii) a Gaussian-based nonlinear thermometer encoding to improve model accuracy. We also present an FPGA-based accelerator for this architecture, targeting low-power edge devices, shown in Figure 4. We incorporate both hardware and software improvements over the prior work.  

# A. Model  

Our objective is to create a hardware-aware, high-accuracy, high-throughput WNN architecture. To accomplish this goal, we enhance the techniques described in Section II with novel algorithmic and architectural improvements.  

1) Counting Bloom Filters: While Bloom filters were used in prior work [21], we augment them to be counting Bloom filters. Bloom filters can only track whether a pattern has been seen; in order to implement bleaching, we need to know how many times each pattern has been encountered.  

A counting Bloom filter is a variant of the Bloom filter which replaces single-bit filter entries with multi-bit counters.  

When an item is added to the filter, it is fed to all $k$ hash functions, and the corresponding counters are incremented. The classical counting Bloom filter allows for items to be added to the array multiple times, and also allows for items to be removed by decrementing counters (although this introduces a risk of false negatives) [27]. Since we do not need element deletion for bleaching, we can modify the counting Bloom filter to eliminate some potential false positives. Rather than incrementing all $k$ counter values, we find the minimum of the accessed counter values, and increment only the counters which have that value. Note that false negatives are still impossible; if a pattern has been seen $i$ times, then the smallest of its corresponding counter values must be at least $i$ .  

As shown in Figure 5, when performing a lookup, a counting Bloom filter returns 1 if the smallest counter value accessed is at least some threshold $b$ ; thus, the possible responses become “possibly seen at least $b$ times” and “definitely not seen $b$ times”.  

2) Hash Function Selection: Bloom filters require multiple distinct hash functions, but do not prescribe what those hash functions should be. Prior work, including Bloom WiSARD [21] [25], used a double-hashing technique based on the MurmurHash [28] algorithm. However, this approach requires many arithmetic operations (e.g. 5 multiplications to hash a 32-bit value), and is therefore impractical in hardware. We identified an alternative approach based on sampling universal families of hash functions which is much less expensive to implement. Thus, while prior work used software-implemented Bloom filters, our design incorporates realistic filters which abide by hardware constraints.  

A universal family of hash functions is a set of functions such that the odds of a hash collision are low in expectation for all functions in the family [29]. Some universal families consist of highly similar functions, which differ only by the choices of constant ”seed” parameters. We considered two such families when designing BTHOWeN.  

The Multiply-Shift hash family [30] is a universal family of non-modulo hash functions which, for an $n$ -bit input size and an $m$ -bit output size, implement the function $h ( x ) = ( a x + b ) \gg ( n - m )$ , where $a$ is an odd $n$ -bit integer, and $b$ is an $( n - m )$ -bit integer. The Multiply-Shift hash function consists of only a few machine instructions, so is easily implemented on a CPU. However, multiplication is a relatively expensive operation in FPGAs, especially when many computations must be performed in parallel.  

By contrast, the H3 family of hash functions [29] requires no arithmetic operations. For an $n$ -bit input $x$ and $m$ -bit output, hash functions in the H3 family take the form:  

$$
h ( x ) = x [ 0 ] p _ { 0 } \oplus x [ 1 ] p _ { 1 } \oplus . . . \oplus x [ n - 1 ] p _ { n - 1 }
$$  

Here, $x [ i ]$ is the $\mathbf { \chi } _ { i } ^ { i }$ ’th bit of $x$ , and $P = \{ p _ { 0 } \ldots p _ { n - 1 } \}$ consists of $n$ random $m$ -bit values. The drawback of the H3 family is that its functions require substantially more storage for parameters when compared to the Multiply-Shift family: nm bits versus just $2 n - m$ .  

![](images/49b4e50669c13e5b2dba62805d8fd89efbbe98acdda4838f14b463dab150e027.jpg)  
Fig. 4. A diagram of the BTHOWeN inference accelerator architecture. We divide Bloom filters into dedicated Hasher and Lookup blocks. The Hasher block computes the H3 hash function on the input data, using a shared set of random hash parameters. The Discriminator block takes hashed data as input, passes it through Lookup units, and performs a popcount on the result, returning a response. The Lookup block contains a LUT, which is accessed using the addresses produced by the hashers, and performs an AND reduction on the results of multiple accesses.  

![](images/83ba1f4bfd3aa651672b3dd37c6d9d0055c9bf8ea01a3dc44d86de6d72b68c50.jpg)  
Fig. 5. An example of a counting Bloom filter with $k = 3$ hash functions. Hashed input $x _ { 1 }$ corresponds to locations containing $\{ 3 , 4 , 2 \}$ ; hashed input $x _ { 2 }$ corresponds to locations containing $\{ 3 , 4 , 3 \}$ . $b \ = \ 3$ in this example (shown by the input to the $\scriptstyle \overbrace { \mathbf { \Lambda } } > = { } ^ { , , , }$ block), so $x _ { 1 }$ produces an output of 0 while $x _ { 2 }$ produces an output of 1.  

In practice, using Bloom filters in a WiSARD model requires many independent filters, each replacing a single RAM node. Each filter in turn requires multiple hash functions. We draw all hash functions from the same universal family, and use $\mathcal { P } = \{ P _ { 0 } . . . P _ { k - 1 } \}$ to represent the random parameters for a filter’s $k$ hash functions.  

For an implementation which uses Multiply-Shift hash functions, many multiplications need to be computed in parallel. This requires a large number of DSP slices on an FPGA. On the other hand, when using H3 hash functions, a large register file is needed for each set of hash parameters $\mathcal { P }$ . However, we observed that sharing $\mathcal { P }$ between Bloom filters did not cause any degradation in accuracy. This effectively eliminates the only comparative disadvantage of the H3 hash function; hence, BTHOWeN uses the H3 hash function with the same $\mathcal { P }$ shared between all filters.  

Cryptographically-secure hash functions such as SHA and MD5 are a poor choice for Bloom filters, as their security features introduce substantial computational overhead.  

3) Implementing Thermometer Encoding: Another enhancement we introduce in BTHOWeN is Gaussian non-linear thermometer encoding. Most prior work using thermometer encodings uses equal intervals between the thresholds. The disadvantage of this approach is that a large number of bits may be dedicated to encoding outlying values, leaving fewer bits to represent small differences in the range of common values.  

For thermometer encoding in BTHOWeN, we assume that each input follows a normal distribution, and compute its mean and standard deviation from training data.For a $t$ -bit encoding, we divide the Gaussian into $t + 1$ regions of equal probability. The values of the divisions between these regions become the thresholds we use for encoding. This provides increased resolution for values near the center of their range.  

# B. Training BTHOWeN  

The process of training a network with the BTHOWeN architecture is shown in Figure 6. Hyperparameters, including the number of inputs in each sample, the number of output classes, details of the thermometer encoding, and configuration information for the Bloom filters, are used to initialize the model.  

During training, samples are presented sequentially to the model. The label of the sample is used to determine which discriminator to train. The input is encoded, passed through the pseudo-random mapping, and presented to the filters in the correct discriminator. Filters hash their inputs and update their corresponding entries.  

![](images/c2e046d91fc0201360e43632de7a5b1d4c600d3f0dfd0ebefbda8eaac4172070.jpg)  
Fig. 6. The training process for BTHOWeN models. Hyperparameters, consisting of the numbers of inputs and categories for the dataset, as well as tunable parameters, are used to construct an “empty” model, where all counter values are 0. Encoded training samples are sequentially presented to the model to update counter values. A validation set is used to select the optimal bleaching threshold $b$ . This threshold is then used to binarize the trained model, replacing counters with binary values. We compare multiple models with different hyperparameters to find targets for implementation.  

After training, the model is evaluated using the validation set at different bleaching thresholds. A binary search strategy is used to select the bleaching threshold $b$ which gives the highest accuracy. The model is then binarized by replacing filter entries less than $b$ with 0, and all other entries with 1. Binarization does not impact model accuracy, and allows counting Bloom filters to be replaced with conventional Bloom filters, which require less memory and are simpler to implement in hardware.  

# C. Inference with BTHOWeN  

Figure 4 shows the design of an accelerator for inference with BTHOWeN WNNs. Since reusing the same random hash parameters for all Bloom filters does not degrade accuracy, we use a central register file to hold the hash parameters. Since all discriminators receive the same inputs, Bloom filters which are at the same index but in different discriminators (e.g. filter 1 in $d _ { 1 }$ and filter 1 in $d _ { 2 }$ ) also receive identical inputs. This means that their hashed values are also identical. It is redundant and inefficient to compute the same hashed values in each discriminator. Instead, we divide the Bloom filters into separate hashing units and lookup units, where hashing units perform H3 hash operations, and lookup filters hold the Bloom filter data and perform AND reductions to determine the filter response. We place the hashing units at the top level of the design, before the discriminators, and broadcast their outputs to all discriminators. Since Bloom filters at the same index across different discriminators have different contents in their RAMs, lookup units can not also be shared across discriminators.  

If the bus bringing data from off-chip has insufficient bandwidth, then the accelerator will finish before the next input is ready. In this case, we can reduce the number of hash units by having each one compute the hashed inputs for multiple lookup units. We store partial results in a large central register until all hashes have been computed for a single set of hash parameters, then pass the hash results to all filters simultaneously, ensuring they operate in lockstep. This strategy allows us to reduce the area of the design without decreasing effective throughput.  

The popcount module counts the number of 1s in the outputs of the filters in a discriminator, and the argmax module determines the index of the discriminator with the strongest response. These are equivalent to the corresponding modules in a conventional WiSARD model.  

Since training with bleaching requires multi-bit counters for each entry in each Bloom filter, it introduces a large amount of memory overhead. For instance, in our experimentation, we found that some models had optimal bleaching values of more than 400. If we used saturating counters large enough to represent this value in the accelerator, it would increase the memory usage of the design by a factor of 9. Since our accelerator is intended for use in low-power edge devices, the advantages of supporting on-chip training do not seem worth the cost.  

# IV. EVALUATION METHODOLOGY  

# A. Hardware Implementation  

Our hardware source is written using Mako-templated SystemVerilog. Mako is a template library for Python which allows the Python interpreter to be run as a preprocessing step for arbitrary languages. This allows for greater flexibility and ease of use than the SystemVerilog preprocessor alone. When Mako is invoked, it generates pure SystemVerilog according to user-specified design parameters.  

We targeted two different Xilinx FPGAs for this project. For most designs, we used the xc7z020clg400-1 FPGA, a small, inexpensive FPGA available in the Zybo Z7 development board, which was used for prior work [31]. For our largest design, we targeted the Kintex UltraScale xcku035-ffva1156- 1-c. Timing, power, and area numbers were obtained from Xilinx Vivado. The choice by the prior work to profile the entire system revealed a major bottleneck in the form of SD card read bandwidth [31]; we are interested in the performance of the accelerator itself. We implement all models with a 100 MHz clock rate, and collect power numbers assuming a $1 2 . 5 \%$ switching rate.  

Hash units produce output at a maximum throughput of 1 hash/cycle. Lookup units can consume hashed inputs at a rate of 1/cycle, and produce output at a rate of $1 / k$ cycles, where $k$ is the number of hash functions associated with a Bloom filter. Therefore, there is no point in having more hashing units than lookup units, and the maximum throughput of the design is $1 / k$ cycles. This throughput could be improved by allowing multiple addresses to be read simultaneously in the lookup units. However, this would greatly increase circuit area, and such a design would likely not be any faster in practice; $k$ is typically small enough that reading data into the accelerator will almost always be the bottleneck.  

At the top level of the design, we use a double-buffered deserialization unit which accumulates input data from the bus until a full sample has been read, then passes the entire sample to the accelerator. This helps enable all hardware units to operate in lockstep, simplifying our state machine logic and verification effort.  

# B. Datasets and Training  

We created models for all classifier datasets discussed in [21]: MNIST [20], Ecoli [32], Iris [33], Letter [34], Satimage [35], Shuttle [36], Vehicle [37], Vowel [38], and Wine [39]. Since our accelerator does not support on-chip training, we implemented the training of models in software. This was done in Python, using the Numba JIT compiler to reduce the runtime of performance-critical functions. We performed a 90- 10 train/validation split on the input dataset, using the former to learn the values in the counting Bloom filters and the latter to set the bleaching threshold.  

# C. WNN Model Sweeping  

There are several model hyperparameters which can be changed to impact the size and accuracy of the model. Increasing the size of the Bloom filters decreases the likelihood of false positives, and thus improves accuracy. However, this greatly increases the model size, and eventually provides diminishing returns for accuracy as false positives become too rare to matter. Increasing the number of input bits to each Bloom filter broadens the space of Boolean functions the filters can learn to approximate, and makes the model size smaller as fewer Bloom filters are needed in total. However, it also increases the likelihood of false positives, since more unique patterns are seen by each filter. Increasing the number of hash functions per Bloom filter can improve accuracy up to a point, but past a certain point actually begins to increase the frequency of false positives [22]. Lastly, increasing the number of bits in the thermometer encoding can improve accuracy at the cost of model size, but again provides diminishing returns as the amount of information each bit conveys decreases.  

In order to identify optimal model hyperparameters, we ran many different configurations in parallel using an automated sweeping methodology. For MNIST, we used 1008 distinct configurations, sweeping all combinations of the hyperparameter settings shown in Table II. For smaller datasets, we explored using 1-16 encoding bits per input, 128-8192 entries per Bloom filter, 1-6 hash functions per filter, and 6-64 input bits per Bloom filter.  

TABLE II HYPERPARAMETERS SWEPT FOR THE CREATION OF BTHOWEN MODELS FOR THE MNIST DATASET   


<html><body><table><tr><td>Hyperparameter</td><td>Values</td></tr><tr><td>Encoding Bits per Input</td><td>1, 2, 3, 4, 5, 6, 7, 8</td></tr><tr><td>InputBitsperBloomFilter</td><td>28,49,56</td></tr><tr><td>EntriesperBloom Filter</td><td>128,256,512,1024，2048,4096,8192</td></tr><tr><td>Hash Functions per Bloom Filter</td><td>1,2,3,4,5, 6</td></tr></table></body></html>  

# D. DNN Model Identification and Implementation  

For each dataset, we trained MLPs that had similar accuracy to our BTHOWeN models. We identified the smallest isoaccuracy MLPs using a hyperparameter sweep. The trained models were then quantized to 8-bit precision to generate a TensorFlow Lite model. Hardware was generated for each MLP using the $\mathrm { \ h { 1 } s { 4 m l } }$ tool [40]. $\mathrm { h } \mathrm { 1 } \mathrm { s } 4 \mathrm { m } \mathrm { 1 }$ takes 4 inputs: (1) the weights generated by TensorFlow (.h5 format), (2) the structure of the model generated by TensorFlow (.json format), (3) the precision to be used for the hardware, and (4) the FPGA part being targeted. It generates $\mathbf { C } { + + }$ code corresponding to the model, and then invokes Xilinx Vivado HLS to generate the hardware design. We modified the generated $\mathbf { C } { + + }$ code such that the I/O interface width matched that of our hardware design for WNNs in order to ensure a fair comparison. We also modified HLS pragmas as needed to ensure that the resultant RTL could fit on the Zybo FPGA. The hardware design generated by Vivado HLS (invoked by hls4ml) was then synthesized and implemented using Xilinx Vivado to obtain area, latency, and power consumption metrics.  

For the MNIST dataset, in addition to MLPs, we compared the BTHOWeN implementation with comparably accurate CNNs based on the LeNet-1 [20] architecture. The LeNet1 implementations generated by $\mathrm { \ h { 1 } s { 4 m l } }$ consume an order of magnitude more area and energy than optimized implementations reported in literature [41]. Therefore, in order to make a more fair comparison, we used the latency and resource numbers for optimized implementations reported by Arish et. al. [41]. We then used the Xilinx Power Estimator (XPE) [42] to get approximate power values for the CNN.  

# V. RESULTS  

# A. Selected BTHOWeN Models  

After performing a hyperparameter sweep, we needed to select one or more trained models for FPGA implementation, balancing tradeoffs between model size and accuracy. For each dataset except for MNIST, there was one model which was very clearly the best, with all more accurate models being many times larger.  

Since MNIST is a more complex dataset, there was no clear single “best” model - instead, we identified “Small”, “Medium”, and “Large” models, which balanced size and accuracy at different points. Our objectives for the three MNIST models were:  

The small model would be comparable in area to the prior FPGA model in [31]   
The medium would be larger, but could still fit on the same FPGA (i.e. the Zybo Z7 board)   
The large model would fit on a mid-size commercial FPGA  

We also experimented with MNIST models using traditional linear thermometer encodings, and observed a $1 2 . 9 \%$ reduction in mean error using the Gaussian encoding.  

The configurations for all the models we selected are shown in Table III.  

TABLE III DETAILS OF THE SELECTED BTHOWEN MODELS   


<html><body><table><tr><td>Model Name</td><td>Bits /Input</td><td>Bits /Filter</td><td>Entries /Filter</td><td>Hashes /Filter</td><td>Size (KiB)</td><td>Test Acc.</td></tr><tr><td>MNIST-Small</td><td>2</td><td>28</td><td>1024</td><td>2</td><td>70.0</td><td>0.934</td></tr><tr><td>MNIST-Medium</td><td>3</td><td>28</td><td>2048</td><td>2</td><td>210</td><td>0.943</td></tr><tr><td>MNIST-Large</td><td>6</td><td>49</td><td>8192</td><td>4</td><td>960</td><td>0.952</td></tr><tr><td>Ecoli</td><td>10</td><td>10</td><td>128</td><td>2</td><td>0.875</td><td>0.875</td></tr><tr><td>Iris</td><td>3</td><td>2</td><td>128</td><td>1</td><td>0.281</td><td>0.980</td></tr><tr><td>Letter</td><td>15</td><td>20</td><td>2048</td><td>4</td><td>78.0</td><td>0.900</td></tr><tr><td>Satimage</td><td>8</td><td>12</td><td>512</td><td>4</td><td>9.00</td><td>0.880</td></tr><tr><td>Shuttle</td><td>9</td><td>27</td><td>1024</td><td>2</td><td>2.63</td><td>0.999</td></tr><tr><td>Vehicle</td><td>16</td><td>16</td><td>256</td><td>3</td><td>2.25</td><td>0.762</td></tr><tr><td>Vowel</td><td>15</td><td>15</td><td>256</td><td>4</td><td>3.44</td><td>0.900</td></tr><tr><td>Wine</td><td>9</td><td>13</td><td>128</td><td>3</td><td>0.422</td><td>0.983</td></tr></table></body></html>  

# B. Comparison with Iso-Accuracy Deep Neural Networks  

Table IV shows FPGA implementation results for BTHOWeN models and iso-accuracy quantized DNNs identified using a hyperparameter sweep across the nine datasets. For the MNIST dataset, the medium BTHOWeN model is only $0 . 3 \%$ less accurate than the MLP, consumes just $1 6 \%$ of the energy of the MLP model, and reduces latency by almost $9 6 \%$ . The MLP uses fewer LUTs and FFs than the medium BTHOWeN model, but also requires DSP blocks and BRAMs on the FPGA. The BTHOWeN model compares even more favorably against CNNs. For example, CNN-1 has an accuracy of $9 4 . 7 \%$ , which is only slightly better than the $9 4 . 3 \%$ accuracy of the medium BTHOWeN model. But even with a pipelined CNN implementation, BTHOWeN consumes less than $0 . 4 \%$ of the energy of the CNN, while reducing latency from 33.6k cycles to just 37.  

As Table IV illustrates, for all datasets except MNIST and Letter, the BTHOWeN model’s hardware implementation consumes less resources (LUTs and FFs) than its MLP counterpart. The reduction in total energy consumption of the BTHOWeN models ranges from $56 . 2 \%$ on Letter to $9 0 . 8 \%$ on Vowel. Reduction in latency ranges from $6 6 . 7 \%$ on Letter to $9 0 . 0 \%$ on Iris.  

Figure 7 summarizes these results, showing the relative latencies, dynamic energies, and total energies of BTHOWeN models compared to DNNs. Overall, BTHOWeN models are significantly faster and more energy efficient than DNNs of comparable accuracy.  

The different-sized models for MNIST, shown in Table IV, provide multiple tradeoff points for energy and accuracy. The Small and Medium models provide good energy efficiency, though the Medium model uses over twice the energy for a $0 . 9 \%$ improvement in accuracy. The Large model does not fit on the Zybo FPGA, so is implemented on a larger FPGA with much higher static power consumption, and is much slower and less energy-efficient. However, if we take advantage of the large number of I/O pins on the large FPGA and implement a 256b bus, energy consumption is reduced by nearly a factor of 4 (the “Large\*” row in Table IV).  

# C. Comparison with Prior Weightless Neural Networks  

Bloom WiSARD, the prior state-of-the-art for WNNs, used Bloom filters to achieve far smaller model sizes than conventional WiSARD models with only slight penalties to accuracy [21]. Results were reported on nine multi-class classifier datasets, which we adopted for our analysis.  

We compared the BTHOWeN models in Table III against the results reported by Bloom WiSARD on all nine datasets, achieving superior accuracy with a smaller model parameter size in all cases. Details are shown in Table V and summarized in Figure 8. On average, our models have $41 \%$ less error with a $51 \%$ smaller model size compared to Bloom WiSARD, which did not incorporate bleaching or thermometer encoding. Our improvements indicate the benefits of these techniques. This comparison is done for software only, since the prior work did not have a hardware implementation. However, we anticipate that our advantage in hardware would be even larger due to our much simpler and more efficient choice of hash function.  

One unusual result is on the Shuttle dataset, for which our model has $\sim 9 9 \%$ less error than prior work. Shuttle is an anomaly-detection dataset in which $80 \%$ of the training data belongs to the “normal” class [36]. We suspect that, since Bloom WiSARD does not incorporate bleaching, the discriminator corresponding to this class became saturated during training.  

# D. Comparison with Prior FPGA Implementation  

In [31], a WNN accelerator for MNIST was implemented on the Zybo FPGA (xc7z020clg400-1) with Vivado HLS. We used this same FPGA at the same frequency $( 1 0 0 ~ \mathrm { { \ M H z } ) }$ . The row with Mode $\ c = ^ { \ d }$ “Hashed WNN” in Table IV shows the implementation results for the prior art accelerator. Its latency and energy consumption are between our small and medium models, but it is much less accurate than even our small model. This accelerator is greatly harmed by its slow memory access, which increases the impact of static power on its energy consumption.  

Exact accelerator latency values were not published. The accelerator reads in one 28-bit filter input per cycle, and uses a 1-bit-per-input encoding, so it takes 28 cycles to read in a 784-bit MNIST sample. Therefore, we use 28 cycles as a lower bound on the time per inference for their design. The energy number in Table IV for [31] is a lower bound based on this cycle count and published power values.  

TABLE IV COMPARISON OF BTHOWEN FPGA MODELS WITH QUANTIZED DNNS OF SIMILAR ACCURACY IMPLEMENTED IN FPGAS. CNNS FOR MNIST ARE LENET-1 VARIATIONS FROM [41]. WNN AND MLP FOR EACH DATA SET IS GROUPED IN NEARBY ROWS FOR EASY COMPARISON.   


<html><body><table><tr><td>Dataset</td><td>Model</td><td>Bus Width</td><td>Cycles per</td><td>Hash Units</td><td>Dyn. Power (Tot. Power)</td><td>Dyn. (Tot. Energy)</td><td>Energy LUTs</td><td>FFs</td><td>(36Kb)</td><td>BRAMs DSPs</td><td>Accuracy</td></tr><tr><td rowspan="8">MNIST</td><td>BTHOWeN-Small</td><td>64</td><td>Inf. 25</td><td>5</td><td>(W) 0.195 (0.303)</td><td>(nJ/Inf.) 48.75 (75.8)</td><td>15756</td><td>3522</td><td>0</td><td>0</td><td>0.934</td></tr><tr><td>BTHOWeN-Medium</td><td>64</td><td>37</td><td>5</td><td>0.386 (0.497)</td><td>142.8 (183.9)</td><td>38912</td><td>6577</td><td>0</td><td>0</td><td>0.943</td></tr><tr><td>BTHOWeN-Large</td><td>64</td><td>74</td><td>6</td><td>3.007 (3.509)</td><td>2225 (2597)</td><td>151704</td><td>18796</td><td>0</td><td>0</td><td>0.952</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BMHPW84-16-10*</td><td>26</td><td>19</td><td>24</td><td>3.028 (3.64)</td><td>20 0 (1335.6)</td><td>158337</td><td>25905</td><td>08</td><td>028</td><td>0.952</td></tr><tr><td>CNN1(LeNet1)[41]</td><td>64</td><td>33615</td><td>-</td><td>0.058 (0.163)</td><td>19497 (54792)</td><td>5753</td><td>3115</td><td>7</td><td>18</td><td>0.947</td></tr><tr><td>CNN 2 (LeNet1)[41]</td><td>64</td><td>33555</td><td></td><td>0.043 (0.148)</td><td>14429 (49661)</td><td>3718</td><td>2208</td><td>5</td><td>10</td><td>0.920</td></tr><tr><td>HashedWNN[31]</td><td>32</td><td>28</td><td></td><td>0.423 (0.528)</td><td>118.4 (147.8)</td><td>9636</td><td>4568</td><td>128.5</td><td>5</td><td>0.907</td></tr><tr><td rowspan="3">Ecoli</td><td>BTHOWeN</td><td>64</td><td>2</td><td>7</td><td>0.012 (0.117)</td><td>0.24 (2.34)</td><td>353</td><td>223</td><td>0</td><td>0</td><td>0.875</td></tr><tr><td>MLP 7-8-8</td><td>64</td><td>14</td><td>_</td><td>0.03 (0.135)</td><td>4.2 (18.9)</td><td>1596</td><td>1615</td><td>0</td><td>0</td><td>0.875</td></tr><tr><td>BTHOWeN</td><td>64</td><td>1</td><td>6</td><td>0.005 (0.109)</td><td>0.05 (1.09)</td><td>57</td><td>90</td><td>0</td><td>0</td><td>0.980</td></tr><tr><td rowspan="2">Iris</td><td>MLP 4-4-3</td><td>64</td><td>10</td><td>-</td><td>0.008 (0.112)</td><td>0.8 (11.2)</td><td>427</td><td>488</td><td>0</td><td>0</td><td></td></tr><tr><td>BTHOWeN</td><td>64</td><td>4</td><td>12</td><td>0.623 (0.738)</td><td>24.92 (29.52)</td><td>21603</td><td>2715</td><td>0</td><td></td><td>0.980</td></tr><tr><td rowspan="2">Letter</td><td>MLP 16-40-26</td><td>64</td><td>26</td><td>-</td><td>0.109 (0.259)</td><td>39.52 (67.34)</td><td>17305</td><td>15738</td><td>0</td><td>0 0</td><td>0.900</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.904</td></tr><tr><td rowspan="2">Satimage</td><td>MLPTH6-W6-N-6</td><td>6464</td><td>55</td><td>24</td><td>0.034 (0.194)</td><td>4.2 5936)</td><td>3777</td><td>1131</td><td>00</td><td>00</td><td>0.88</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Shuttle</td><td>MTHOW4-N</td><td>6464</td><td>214</td><td>3-</td><td>0.018 (0.123</td><td>0.36 (2.62)</td><td>593</td><td>121</td><td>00</td><td>00</td><td>0.999</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Vehicle</td><td>MLH08-164</td><td>6464</td><td>55</td><td>18</td><td>0.0248 (0.123)</td><td>13.9 (7.25)</td><td>1784</td><td>5975</td><td>00</td><td>00</td><td>0.762</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Vowel</td><td>MLTH0We-11</td><td>6464</td><td>218</td><td>2.</td><td>0.040 (0.145)</td><td>1. (.31.5)</td><td>1559</td><td>756</td><td>00</td><td>00</td><td>0.900</td></tr><tr><td></td><td></td><td></td><td></td><td>0.012 (0.117)</td><td>0.36 (3.51)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Wine</td><td>BTHOWeN</td><td>64</td><td>3</td><td>9</td><td>0.026 (0.131)</td><td>3.64 (18.34)</td><td>585</td><td>239</td><td>0</td><td>0</td><td>0.983</td></tr><tr><td>MLP13-10-3</td><td>64</td><td>14</td><td>-</td><td></td><td></td><td>1836</td><td>1832</td><td>0</td><td>0</td><td>0.983</td></tr></table></body></html>  

![](images/5fa5026378a0451505c6fae285678ca01fd8f578385171a664c601b6055abfbf.jpg)  
Fig. 7. The relative latencies and energies of BTHOWeN models versus iso-accuracy DNNs. For MNIST, our Medium model is compared against the MNIST MLP model, and our Large (64b bus) model is compared against CNN 1. The results of the comparison with CNN 1 are not used in the computation of the average, since the Large model uses a different FPGA. We implemented baseline MLPs and BTHOWeNmodels at ${ 1 0 0 } \mathrm { M H z }$ and obtained metrics from Xilinx tools.  

TABLE V ACCURACY AND MODEL SIZE (IN KIB) OF PROPOSED BTHOWEN VS BEST MEMORY-EFFICIENT PRIOR WORK (BLOOM WISARD) [21]   


<html><body><table><tr><td>Model Name</td><td>Accuracy (Bloom WiSARD)</td><td>Accuracy (This work)</td><td>Size (Bloom WiSARD)</td><td>Size (This work)</td></tr><tr><td>MNIST-Small</td><td rowspan="3">0.915</td><td>0.934</td><td rowspan="3">819</td><td>70.0</td></tr><tr><td>MNIST-Medium</td><td>0.943</td><td>210</td></tr><tr><td>MNIST-Large</td><td>0.952</td><td>960</td></tr><tr><td>Ecoli</td><td>0.799</td><td>0.875</td><td>3.28</td><td>0.875</td></tr><tr><td>Iris</td><td>0.976</td><td>0.980</td><td>0.703</td><td>0.281</td></tr><tr><td>Letter</td><td>0.848</td><td>0.900</td><td>91.3</td><td>78.0</td></tr><tr><td>Satimage</td><td>0.851</td><td>0.880</td><td>12.7</td><td>9.00</td></tr><tr><td>Shuttle</td><td>0.868</td><td>0.999</td><td>3.69</td><td>2.63</td></tr><tr><td>Vehicle</td><td>0.662</td><td>0.762</td><td>4.22</td><td>2.25</td></tr><tr><td>Vowel</td><td>0.876</td><td>0.900</td><td>6.44</td><td>3.44</td></tr><tr><td>Wine</td><td>0.926</td><td>0.983</td><td>2.28</td><td>0.422</td></tr></table></body></html>  

Our implementation has significant differences which contribute to BTHOWeN’s superior accuracy and efficiency:  

The prior accelerator used a simple hash-table-based encoding scheme which had explicit hardware for collision detection; we use an approach based on counting Bloom filters which does not need collision detection. • Models for the prior accelerator did not incorporate bleaching or thermometer encoding; instead, they used a simple 1-bit encoding based on comparison with a parameter’s mean value. We use counting Bloom filters to enable bleaching.  

![](images/30a8c5c983c0b674e0f5de079c1ad96125639126add5302ffef32b33b711cb9e.jpg)  
Fig. 8. The relative errors and model sizes of the models shown in Table III versus Bloom WiSARD [21]. BTHOWeN outperforms the prior work on all nine datasets in both accuracy and model size. For the MNIST dataset, our MNIST-Medium model was used for comparison.  

Since the prior accelerator [31] did not incorporate bleaching, training did not require multi-bit counters, making it inexpensive to support.  

# E. Model Tradeoff Analysis  

We use MNIST as an illustrative example of the tradeoffs present in model selection. Figure 9 presents the results obtained from sweeping over the MNIST dataset with the configurations presented in Table II. In the first four subplots of Figure 9, we vary one hyperparameter of the model, respectively, the number of bits used to encode each input, the number of inputs to each Bloom filter, the number of entries in each filter, and the number of distinct hash functions for each filter. We show four lines: three of them represent the Small, Medium, and Large models where only the specified hyperparameter was varied, while the fourth represents the best model with the given value for the hyperparameter.  

We see diminishing returns as the number of encoding bits per input and the number of entries per Bloom filter increase. The Small and Medium models rapidly lose accuracy as the number of inputs per filter increases, but the Large model, with its large filter LUTs, is able to handle 49 inputs per filter without excessive false positives. These results align with the theoretical behaviors discussed earlier.  

One surprising result was that, although there was a slight accuracy increase going from 1 hash function per filter to 2, continuing to increase this had minimal impact. In theory, we would expect that continuing to increase this value would eventually result in a loss of accuracy due to high false positive rates. One explanation for this is that the BTHOWeN model reports the index of the class with the strongest response; since a higher false positive rate would impact the response of all classes, the predicted class should remain unaffected as long as the increase is proportional. Another observation, shown in the fifth subplot of Figure 9, is that the optimal bleaching value $b$ increases to compensate for the larger number of hash functions. This plot shows variants of the Small model with up to 128 hash functions per Bloom filter. When $b$ is fixed at 16, accuracy collapses, but when the optimal value is chosen using the same binary search strategy we use normally, it is better able to compensate. This provides a good example of how bleaching improves the robustness of BTHOWeN.  

The last subplot shows the most accurate MNIST model we were able to obtain with a given maximum model size. We notice diminishing returns as model size increases. It is evident that in order to exceed $9 6 \%$ accuracy with reasonable model sizes, additional algorithmic improvements will be needed.  

# VI. CONCLUSION  

While most machine learning research centers around DNNs, we explore an alternate neural model, the Weightless Neural Network, for edge inference. We incorporate enhancements such as counting Bloom filters, inexpensive H3 hash functions and a Gaussian-based non-linear thermometer encoding into the WiSARD weightless neural model, improving state-of-the-art WNN MNIST accuracy [21] from $9 1 . 5 \%$ to $9 5 . 2 \%$ . The proposed BTHOWeN architecture is compared to state-of-the-art weightless models as well as MLPs and CNNs of similar accuracy. An FPGA accelerator for BTHOWeN is also presented. Compared to prior WNNs, BTHOWeN reduces error by $41 \%$ and model size by $5 1 \%$ across nine datasets. Compared to iso-accuracy MLP models, BTHOWeN consumes $\sim 2 0 \%$ of the total energy while reducing latency by $\sim 8 5 \%$ . Energy/latency improvements over CNNs are even larger, although CNNs have higher accuracy.  

There are many opportunities for future work in this domain. There are algorithmic improvements we would like to explore, including weightless convolutional neural networks, better input remapping, and converting pretrained DNNs to WNNs. Preliminary experiments suggest that backpropagation-based training approaches can significantly improve WNN model accuracy, making them feasible for broader applications.  

We believe that WNNs hold substantial promise for inference on the edge. While WNNs have historically trailed in accuracy to DNNs, algorithmic improvements such as bleaching demonstrate that accuracy and efficiency can go up with enhanced architectures and training techniques. The low latency and low energy benefits that can be obtained from WNNs warrant further research in this area.  

# REFERENCES  

[1] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. FeiFei, “Imagenet large scale visual recognition challenge,” International Journal of Computer Vision, vol. 115, pp. 211–252, 4 2015.  

![](images/00efd76b6ff9b0c3d23df4869306a1090e205c6702fb14174615dab42cbcac8b.jpg)  
Fig. 9. Sweeping results for MNIST with the configurations described in Table II, showing the impact of (a) the number of bits used to encode each input on accuracy, (b) the number of inputs to each Bloom filter on accuracy, (c) the number of entries in the LUTs of each Bloom filter on accuracy, (d) the number of distinct hash functions per Bloom filter on accuracy, and (e) large numbers of hash functions on accuracy with a fixed vs. varying bleaching value. Subfigure (f) shows the most accurate model which we could obtain under a given maximum model size.  

[2] A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer, “A survey of quantization methods for efficient neural network inference,” 2021.   
[3] I. Aleksander, M. De Gregorio, F. Franc¸a, P. Lima, and H. Morton, “A brief introduction to weightless neural systems,” in 17th European Symposium on Artificial Neural Networks (ESANN), 04 2009, pp. 299– 305.   
[4] V. Sze, H. Yu-Chen, J. Tien-Yang, and J. Emer, “Efficient processing of deep neural networks: A tutorial and survey,” 2017. [Online]. Available: https://arxiv.org/pdf/1703.09039.pdf   
[5] D. Fick and M. Henry, “Mythic AI’s Presentation at Hot Chips 2018, Accessed November 22, 2021,” 2018. [Online]. Available: https://www.mythic-ai.com/mythic-hot-chips-2018/   
[6] I. Aleksander, W. Thomas, and P. Bowden, “WISARD·a radical step forward in image recognition,” Sensor Review, vol. 4, no. 3, pp. 120–124, 1984. [Online]. Available: https://www.emerald.com/insight/ content/doi/10.1108/eb007637/full/html   
[7] V. Vapnik and A. Chervonenkis, On the uniform convergence of relative frequencies of events to their probabilities. Springer, 01 2015, pp. 11–30.   
[8] H. Carneiro, C. Pedreira, F. Fran¸ca, and P. Lima, “The exact vc dimension of the wisard n-tuple classifier,” Neural Computation, pp. 1–32, 11 2018.   
[9] T. Ludermir, A. de Carvalho, A. Braga, and M. Souto, “Weightless neural models: A review of current and past works,” Neural Computing Surveys, vol. 2, pp. 41–61, 01 1999.   
[10] R. Rohwer and M. Morciniec, “The theoretical and experimental status of the $n$ -tuple classifier,” Neural Netw., vol. 11, no. 1, p. 1–14, Jan. 1998.   
[11] I. Wickert and F. Franc¸a, “Validating an unsupervised weightless perceptron,” in Proceedings of the 9th International Conference on Neural Information Processing, 2002. ICONIP ’02., 12 2002, pp. 537 – 541 vol.2.   
[12] D. O. Cardoso, D. Carvalho, D. S. F. Alves, D. F. P. de Souza, H. C. C. Carneiro, C. E. Pedreira, P. M. V. Lima, and F. M. G. Franc¸a, “Financial credit analysis via a clustering weightless neural classifier,” Neurocomputing, vol. 183, pp. 70–78, 2016.   
[13] A. Bacellar, B. Goldstein, V. Ferreira, L. Santiago, P. Lima, and F. Fran¸ca, “Fast deep neural networks convergence using a weightless neural model,” in ESANN, 10 2020.   
[14] D. Kang, “Accelerating queries over unstructured data with ml,” in CIDR, 2021.   
[15] Simone, “TinyML or Arduino and STM32: Convolutional Neural Network (CNN) Example, Accessed Nov 22, 2021,” 2020. [Online]. Available: https://eloquentarduino.github.io/2020/11/ tinyml-on-arduino-and-stm32-cnn-convolutional-neural-network-examp   
[16] B. P. A. Grieco, P. M. V. Lima, M. De Gregorio, and F. M. G. Fran¸ca, “Producing pattern examples from ”mental” images,” Neurocomput., vol. 73, no. 7-9, pp. 1057–1064, Mar. 2010. [Online]. Available: http://dx.doi.org/10.1016/j.neucom.2009.11.015   
[17] D. Carvalho, H. Carneiro, F. Fran¸ca, and P. Lima, “B-bleaching $\because$ Agile overtraining avoidance in the wisard weightless neural classifier,” in ESANN, 04 2013.   
[18] W. S. McCulloch and W. Pitts, “A logical calculus of the ideas immanent in nervous activity,” The Bulletin of Mathematical Biophysics, vol. 5, no. 4, pp. 115–133, 1943.   
[19] M. Courbariaux, I. Hubara, D. Soudry, R. El-Yaniv, and Y. Bengio, “Binarized neural networks: Training deep neural networks with weights and activations constrained to $+ 1$ or -1,” 2016.   
[20] Y. LeCun and C. Cortes, “MNIST handwritten digit database,” http://yann.lecun.com/exdb/mnist/, 2010. [Online]. Available: http: //yann.lecun.com/exdb/mnist/   
[21] L. S. de Arau´jo, L. D. Verona, F. M. Rangel, F. F. de Faria, D. S. Menasch´e, W. Caarls, M. Breternitz, S. Kundu, P. M. V. Lima, and F. M. G. Fran¸ca, “Memory efficient weightless neural network using bloom filter,” in ESANN, 2019.   
[22] B. H. Bloom, “Space/time trade-offs in hash coding with allowable errors,” Commun. ACM, vol. 13, no. 7, p. 422–426, Jul. 1970. [Online]. Available: https://doi.org/10.1145/362686.362692   
[23] K. Gopinathan and I. Sergey, “Certifying certainty and uncertainty in approximate membership query structures,” in Computer Aided Verification, S. K. Lahiri and C. Wang, Eds. Cham: Springer International Publishing, 2020, pp. 279–303.   
[24] M. Breternitz, G. H. Loh, B. Black, J. Rupley, P. G. Sassone, W. Attrot, and Y. Wu, “A segmented bloom filter algorithm for efficient predictors,” in 2008 20th International Symposium on Computer Architecture and High Performance Computing. IEEE, 2008, pp. 123–130.   
[25] L. Santiago, L. Verona, F. Rangel, F. Firmino, D. S. Menasche´, W. Caarls, M. Breternitz Jr, S. Kundu, P. M. Lima, and F. M. Franc¸a, “Weightless neural networks as memory segmented bloom filters,” Neurocomputing, vol. 416, pp. 292–304, 2020.   
[26] A. Kappaun, K. Camargo, F. Rangel, F. Firmino, P. M. V. Lima, and J. Oliveira, “Evaluating binary encoding techniques for wisard,” in 2016 5th Brazilian Conference on Intelligent Systems (BRACIS), 2016, pp. 103–108.   
[27] K. Kim, Y. Jeong, Y. Lee, and S. Lee, “Analysis of counting bloom filters used for count thresholding,” Electronics, vol. 8, no. 7, 2019. [Online]. Available: https://www.mdpi.com/2079-9292/8/7/779   
[28] A. Appleby, “Murmurhash3,” https://github.com/aappleby/smhasher, 2016.   
[29] J. Carter and M. N. Wegman, “Universal classes of hash functions,” Journal of Computer and System Sciences, vol. 18, no. 2, pp. 143–154, 1979. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/0022000079900448   
[30] M. Dietzfelbinger, T. Hagerup, J. Katajainen, and M. Penttonen, “A reliable randomized algorithm for the closest-pair problem,” Journal of Algorithms, vol. 25, no. 1, pp. 19–51, 1997. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0196677497908737   
[31] V. C. Ferreira, A. S. Nery, L. A. J. Marzulo, L. Santiago, D. Souza, B. F. Goldstein, F. M. G. Fran¸ca, and V. Alves, “A feasible fpga weightless neural accelerator,” in 2019 IEEE International Symposium on Circuits and Systems (ISCAS), 2019, pp. 1–5.   
[32] K. Nakai, “Ecoli data set.” [Online]. Available: https://archive.ics.uci. edu/ml/datasets/ecoli   
[33] R. Fisher, “Iris data set.” [Online]. Available: https://archive.ics.uci.edu/ ml/datasets/iris   
[34] D. J. Slate, “Letter recognition data set.” [Online]. Available: https://archive.ics.uci.edu/ml/datasets/letter+recognition   
[35] A. Srinivasan, “Statlog (landsat satellite) data set.” [Online]. Available: https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)   
[36] J. Catlett, “Statlog (shuttle) data set.” [Online]. Available: https: //archive.ics.uci.edu/ml/datasets/Statlog $^ +$ (Shuttle)   
[37] P. Mowforth and B. Shepherd, “Statlog (vehicle silhouettes) data set.” [Online]. Available: https://archive.ics.uci.edu/ml/datasets/Statlog+ (Vehicl $^ { + }$ Silhouettes)   
[38] D. Deterding, M. Niranjan, and T. Robinson, “Connectionist bench (vowel recognition - deterding data) data set.” [Online]. Available: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+ (Vowel+Recognition $^ +$ - $^ +$ Deterding+Data)   
[39] S. Aeberhard, “Wine data set.” [Online]. Available: https://archive.ics. uci.edu/ml/datasets/wine   
[40] J. Duarte, S. Han, P. Harris, S. Jindariani, E. Kreinar, B. Kreis, J. Ngadiuba, M. Pierini, R. Rivera, N. Tran, and Z. Wu, “Fast inference of deep neural networks in FPGAs for particle physics,” Journal of Instrumentation, vol. 13, no. 07, pp. P07 027–P07 027, jul 2018. [Online]. Available: https://doi.org/10.1088/1748-0221/13/07/p07027   
[41] A. S., S. Sinha, and S. K.G., “Optimization of convolutional neural networks on resource constrained devices,” in 2019 IEEE Computer Society Annual Symposium on VLSI (ISVLSI), 2019, pp. 19–24.   
[42] “Xilinx Power Estimator (XPE),” 2021. [Online]. Available: https: //www.xilinx.com/products/technology/power/xpe.html  