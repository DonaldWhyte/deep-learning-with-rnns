<!-- .slide: data-background="images/books_opened.jpg" class="background" -->

<h2>Deep Learning with Recurrent Neural Networks</h2>
<h4>In Python</h4>
<p>
    <a href="http://donaldwhyte.co.uk">Donald Whyte</a>
    / <a href="http://twitter.com/donald_whyte">@donald_whyte</a>
    <br />
    <a href="http://donaldwhyte.co.uk">Alejandro Saucedo</a>
    / <a href="http://twitter.com/AxSaucedo">@AxSaucedo</a><br/>
  <br />
</p>
<p>

[NEXT]
<!-- .slide: data-background="images/books_opened.jpg" class="background" -->

### About Us

<table class="bio-table">
  <tr>
    <td>![portrait](images/alejandro.jpg)</td>
    <td>![portrait](images/donald.jpg)</td>
  </tr>
  <tr>
    <td><span style="color: white; font-weight: bold">Alejandro Saucedo</span></td>
    <td><span style="color: white; font-weight: bold">Donald Whyte</span></td>
  </tr>
</table>

[NEXT]
<!-- .slide: data-background="images/books_opened.jpg" class="background large" -->

Create an AI author.

[NEXT]
<!-- .slide: data-background="images/books_opened.jpg" class="background smallquote" -->

> Gradually drawing away from the rest, two of the combatants are striving; each devoting every nerve, every energy, to the overthrow of the other.
>
> But each attack is met by counter attack, each terrible swinging stroke by the crash of equally hard wood or the dull slap of tough hide shield opposed in parry.
>
> More men are down, about even numbers on each side, these two combatants strive on.

[NEXT]
<!-- .slide: data-background="images/books_opened.jpg" class="background" -->
Create a neural network that can write novels.

Use 30,000 English novels to train the network.


[NEXT]
<!-- .slide: data-background="images/books_opened.jpg" class="background smallest" -->

<div class="left-col">
  <pre><code class="two-col-code python hljs"># ONE

import tensorflow as tf
from tensorflow.contrib import layers, rnn
import os
import time
import math
import numpy as np
tf.set_random_seed(0)

# model parameters
SEQLEN = 30
BATCHSIZE = 200
ALPHASIZE = 89
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001
dropout_pkeep = 0.8

codetext, valitext, bookranges = load_data()

# the model
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X')
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)
# expected outputs
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')
Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)
# input state
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')

# hidden layers
cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
multicell = rnn.MultiRNNCell(cells, state_is_tuple=False)
  </code></pre>
</div>

<div class="right-col">
  <pre><code class="two-col-code python hljs"># TWO

Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
H = tf.identity(H, name='H')

# Softmax layer implementation
Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])
Ylogits = layers.linear(Yflat, ALPHASIZE)
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)
loss = tf.reshape(loss, [batchsize, -1])
Yo = tf.nn.softmax(Ylogits, name='Yo')
Y = tf.argmax(Yo, 1)
Y = tf.reshape(Y, [batchsize, -1], name="Y")
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Init for saving models
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

# init
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

# train on one minibatch at a time
for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=10):
    feed_dict = {X: x, Ye: ye, Hin: istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
    _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)

    if step // 10 % _50_BATCHES == 0:
        saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
        print("Saved file: " + saved_file)

    istate = ostate
    step += BATCHSIZE * SEQLEN
  </code></pre>
</div>

<div class="clear-col"></div>

<!-- .element style="color: white;" -->
75 lines of Tensorflow code!

[NEXT]
#### Gutenberg Datatset

Contains 34,000 English novels.

**https://www.gutenberg.org/**

_note_
Project Gutenberg offers over 54,000 free eBooks: Choose among free epub books,
free kindle books, download them or read them online. You will find the world's
great literature here, especially older works for which copyright has expired.
We digitized and diligently proofread them with the help of thousands of
volunteers.

[NEXT]
![novel_preprocessing](images/novel_preprocessing_intro.svg)

_note_
1. merge all 30000 novels into a single text document
2. load single document as a flat sequence of characters
3. filter out characters we don't care about
4. map each char to an integer
  - integer decides which _input value_ is set to 0

**Result**: sequence of integers

_notes_
Emphasise the fact that you load all of the textual data in as integer-coded
chars. All documents are flattened into a single large sequence.


[NEXT]
### Best AI Author Prize

![prize](images/prize.jpg)

_note_
We're also running a workshop later today where we guide you through the code
from this talk so you can build your own AI author.

We're awarding prizes to the best performing models, so be sure to come along
to that!


[NEXT SECTION]
## 1. Traditional Supervised Learning

Use labelled historical data to predict future outcomes

[NEXT]
Given some input data, predict the correct output

![shapes](images/shapes.svg)

What **features** of the input tell us about the output?

[NEXT]
### Feature Space

* A feature is some property that describes raw input data
* Features represented as a vector in **feature space**
* **Abstract** complexity of raw input for easier processing

![classification](images/classification-points.png)

_note_
In this case, we have 2 features, so inputs are 2D vector that lie in
a 2D feature space.

[NEXT]
### Classification

<div class="left-col">
  <ul>
    <li>Training data is used to produce a model</li>
    <li>$f(x̄) = mx̄ + c$</li>
    <li>Model divides feature space into segments</li>
    <li>Each segment corresponds to one <strong>output class</strong></li>
  </ul>
</div>

<div class="right-col">
  ![classification_small](images/classification-line.png)
</div>

<div class="clear-col"></div>

[NEXT]
Use trained model to predict outcome of new, unseen data.

[NEXT]
### Example

<div class="left-col">
  ![classification_small](images/classification-line.png)
</div>

<div class="right-col">
  <br />
  <br />
  <p>$x̄ = (area, perimeter)$</p>
  <p>$m = (1, -3.5)$</p>
  <p>$c = 0$</p>
</div>

[NEXT]
Using this model, let's predict what shape an object is.

| **Feature** |  **Value** |
| ----------- | ---------- |
| Area        | 3          |
| Perimeter   | 1          |

[NEXT]
<div class="left-col">
  ![classification_small](images/classification-newinput.png)
</div>

<div class="right-col">
  <p>$x̄ = (3, 1)$</p>
  <p>$f(x̄) = 1(3) + -3.5(1) + 0$</p>
  <p>$f(x̄) = 3 - 3.5 + 0$</p>
  <p>$f(x̄) = -0.5$</p>
  <p>$-0.5 < 0$</p>
</div>

<div class="clear-col"></div>
$-0.5 < 0$ means left side of the line.

Input shape is a **triangle**.

_note_
Point out that this same technique can used to predicting continuous, numerical
values too (like how much house prices will cost, or how much a stock's price
will go up or down).

[NEXT]
### The Hard Part

Learning $m$ and $c$.

[NEXT]
Can this be used to learn how to write novels?

[NEXT]
**No.**

[NEXT]
Generating coherent text requires memory of what was written previously.

[NEXT]
> <span style="font-weight: bold; color: red">Valentin's</span> favourite
> <span style="font-weight: bold; color: blue">drink</span> is
> <span style="font-weight: bold; color: blue">beer.</span>
> <span style="font-weight: bold; color: red">He</span> likes
> <span style="font-weight: bold; color: blue">lagers</span> the most.

<table>
  <tr>
    <th>Male Person</th>
    <td><span style="font-weight: bold; color: red">Valentin, he</span></td>
  </tr>
  <tr>
    <th>Drinks</th>
    <td><span style="font-weight: bold; color: blue">drink, beer, lagers</span></td>
  </tr>
  <tr>
  </tr>
</table>

_note_
Other issues with traditional machine learning:

* Does not scale to large numbers of input features
* Relies on you to break raw input data into a small set of useful features
* Good feature engineering requires in-depth domain knowledge and time


[NEXT SECTION]
## 2. Deep Neural Networks

[NEXT]
Deep neural nets can learn to patterns in complex data, like language.

We can encode memory into the algorithm.

_note_
We can encode memory into the algorithm, allowing us to generate more
coherent novels that _remember_ what was previously written.

[NEXT]
Just use the raw input data.

Our training data is the raw text of existing novels.

No need for for manual feature extraction.

[NEXT]
### The Mighty Perceptron

* Equivalent to the straight line equation from before
* Linearly splits feature space
* Modeled after a neuron in the human brain

[NEXT]
![perceptron](images/perceptron.svg)

[NEXT]
### Perceptron Definition

For `n` features, the perceptron is defined as:

* `n`-dimensional weight vector `w`
* bias scalar `b`
* activation function `f(s)`

[NEXT]
|                     |                                            |
| ------------------- | ------------------------------------------ |
| Input               | $x = \left(x_0, x_1, \cdots, w_n\right)$   |
| Weights             | $w = \left(w_0, w_1, \cdots, w_n\right)$   |
| Bias                | $b$                                        |
| Weighted Sum        | $\left(\sum_{i=0}^{n} {w_ix_i}\right) + b$ |
| Activation Function | $f(s)$                                     |


[NEXT]
### Activation Function

Simulates the 'firing' of a physical neuron.

Takes the weighted sum and squashes it into a smaller range.

[NEXT]
### Sigmoid Function

* Squashes perceptron output into range [0,1]
* Used to learn weights (`w`)

![sigmoid_function](images/sigmoid-function.png)

_note_
We'll find having a continuous activation function is very useful when we want
to learn the values of the perceptron weights.

[NEXT]
How do we learn `w` and `b`?

[NEXT]
### Perceptron Learning Algorithm

Algorithm which learns correct weights and bias

Use training dataset to incrementally train perceptron

Guaranteed to create line that divides output classes

(if data is linearly separable)<!-- .element class="small" -->

_note_
Details of the algorithm are not covered here for brevity.

Training dataset, which is a collection of known input/output pairs
(typically produced by humans manually labelling input).

[NEXT]
![perceptron_learning](images/perceptron_learning1.png)

[NEXT]
![perceptron_learning](images/perceptron_learning2.png)

[NEXT]
![perceptron_learning](images/perceptron_learning3.png)

[NEXT]
![perceptron_learning](images/perceptron_learning4.png)

[NEXT]
### Representing Text

Make the input layer represent:

* a single word
* or a single character

Use the input to word/char to predict the next.

[NEXT]
We will use characters as the inputs.

_note_
There are pros and cons with either representation. Both techniques use the
same principle. You use the input token to _predict_ the next token.

Both words and characters are valid ways of representing text in neural
networks. Representing the input as characters is an easier approach and for
many tasks, actually results in better performing networks.

We don't have time to go into more detail in this talk, but feel free to ask
us for more details afterwards.

[NEXT]
![char_perceptron](images/char_perceptron1.svg)

[NEXT]
![char_perceptron_filled](images/char_perceptron2.svg)

<table class="small-table"><tr>
  <td><strong>Input:</strong> <span style="color: blue">b</span></td>
  <td><strong>Predicted char:</strong> <span style="color: red">?</span></td></tr>
  <tr><td><strong>Current sentence:</strong> <span style="color: blue">b</span><span style="color: red">?</span></td></tr>
</table>

[NEXT]
![char_perceptron_filled](images/char_perceptron3.svg)

<table class="small-table"><tr>
  <td><strong>Input:</strong> <span style="color: blue">b</span></td>
  <td><strong>Predicted char:</strong> <span style="color: red">a</span></td></tr>
  <tr><td><strong>Current sentence:</strong> <span style="color: blue">b</span><span style="color: red">a</span></td></tr>
</table>

[NEXT]
![char_perceptron_filled](images/char_perceptron4.svg)

<table class="small-table"><tr>
  <td><strong>Input:</strong> <span style="color: blue">a</span></td>
  <td><strong>Predicted char:</strong> <span style="color: red">d</span></td></tr>
  <tr><td><strong>Current sentence:</strong> <span style="color: blue">ba</span><span style="color: red">d</span></td></tr>
</table>

[NEXT]
...

[NEXT]
![char_perceptron_filled](images/char_perceptron_n.svg)

<table class="small-table"><tr>
  <td><strong>Input:</strong> <span style="color: blue">e</span></td>
  <td><strong>Predicted char:</strong> <span style="color: red">d</span></td></tr>
  <tr><td><strong>Current sentence:</strong> <span style="color: blue">ball games were often playe</span><span style="color: red">d</span></td></tr>
</table>


[NEXT]
### Problem

Single perceptrons are straight line equations.

Produce a single output.

Need a *network* of neurons to output the full one-hot vector.

[NEXT]
### Neural Networks

Uses *many* perceptrons to:

* learn patterns in complex data, like language
* produce the multiple outputs required for text prediction

[NEXT]
![nn_chars_filled_in](images/nn_chars_filled_in.svg)

[NEXT]
![nn_chars_filled_in](images/deep_nn_chars_filled_in.svg)

[NEXT]
|            |                                     |
| ---------- | ----------------------------------- |
| **Input**  | $V$ nodes, a single one-hot vector  |
| **Hidden** | multiple percetrons                 |
| **Output** | $V$ nodes, a single one-hout vector |

where $V$ is the number of characters.

_note_
The hidden layers is where all the smarts comes in. I could spend days
discussing how to choose the number of hidden layers and nodes in each
layer.

It depends on so many factors. The number of input features, the distribution
of inputs across feature space.

[NEXT]
### Neuron Connectivity

* Each layer is **fully connected** to the next
* All nodes in layer $l$ are connected to nodes in layer $l + 1$
* Every single connection has a weight

_note_
Standard neural network architectures make each layer fully connected
to the next.

[NEXT]
Produces multiple **weight matrices**

One for each layer

![weight_matrix](images/weight_matrix.svg)

_note_
Weight matrix produced using the following Latex equation:
W = \begin{bmatrix} w_{00} & w_{01} & \cdots & w_{0n} \\ w_{10} & w_{11} & \cdots & w_{1n} \\ \vdots & \vdots & \vdots & \vdots \\ w_{m0} & w_{m1} & \cdots & w_{mn} \end{bmatrix}

[NEXT]
### Training Neural Networks

Learn the weight matrices!

[NEXT]
Optimisation problem.

[NEXT]
### Gradient Descent Optimiser

Keep adjusting the weights of each hidden layer

In such a way that we minimise incorrect predictions

[NEXT]
TODO: loss function

[NEXT]
![gradient_descent](images/gradient_descent_cropped.gif)

Uses **derivatives** of activation functions to adjust weights

_note_

We can describe the principle behind gradient descent as “climbing down a
hill” until a local or global minimum is reached.

We use the **derivatives** of each neuron's activation function to determine
the direction of the error. The direction of error is the gradient.

At each step, we take a step into the **opposite** direction of the gradient.
That is, we adjust the weights so we have _less_ error in the next step.

This is why we typically use an activation function like sigmoid. Sigmoid
provides a continuous output. It is fast and easy to compute the derivative of
continuous functions. If we used a discrete activiation function, it would be
much harder to run gradient descent.

The step size is determined by the value of the **learning rate** as well
as the slope of the gradient.

Source of diagram: https://medium.com/onfido-tech/machine-learning-101-be2e0a86c96a

[NEXT]
### Backpropagation

* Equivalent to gradient descent
* *The* training algorithm for neural networks
* For each feature vector in the training dataset, do a:
  1. forward pass
  2. backward pass

_note_
Backpropagation is the workhorse of neural network training. Some
variation of this algorithm is almost always used to train nets.

For a data point in our training dataset, we run two steps.

Visualisation of learning by backpropagation:
http://www.emergentmind.com/neural-network

[NEXT]
### Forward Pass
![backprop_forward_pass](images/ffnn_nclass.svg)

_note_
1. Start with random weights
2. Feed input feature vector to input layer
3. Let the first layer evaluate their activation using
4. Feed activation into next layer, repeat for all layers
5. Finally, compute output layer values

[NEXT]
### Forward Pass
![backprop_forward_pass](images/backprop_forwardpass_0.svg)

[NEXT]
### Forward Pass
![backprop_forward_pass](images/backprop_forwardpass_1.svg)

[NEXT]
### Forward Pass
![backprop_forward_pass](images/backprop_forwardpass_2.svg)

[NEXT]
### Forward Pass
![backprop_forward_pass](images/backprop_forwardpass_3.svg)

[NEXT]
### Backward Pass
![backprop_back_pass](images/backprop_backpass_0.svg)

_note_
TODO: loss function -- add a loss function

1. Compare the target output to the actual output
  - calculate the errors of the output neurons
2. Calculate weight updates associated with output neurons using perceptron learning principle
  - same adjustments as the ones made in the Perceptron Algorithm)
3. For each output neuron, propagate values back to the previous layer
4. Calculate weight updates associated with hidden neurons using perceptron learning principle
5. Update weights, then repeat from step 1 (performing another forward and backward pass) until the weight values converge

[NEXT]
### Backward Pass
![backprop_back_pass](images/backprop_backpass_1.svg)

[NEXT]
### Backward Pass
![backprop_back_pass](images/backprop_backpass_2.svg)

[NEXT]
### Backward Pass
![backprop_back_pass](images/backprop_backpass_3.svg)

[NEXT]
After training the network, we obtain weights which minimise prediction error.

Predict next character by running the last character through the
**forward pass** step.

[NEXT]
### However...

> <span style="font-weight: bold; color: red">Valentin's</span> favourite
> <span style="font-weight: bold; color: blue">drink</span> is
> <span style="font-weight: bold; color: blue">beer.</span>
> <span style="font-weight: bold; color: red">He</span> likes
> <span style="font-weight: bold; color: blue">lagers</span> the most.

Network still has <strong>no memory of past characters</strong>.

_note_
We need to know what the last _N_ characters were to effectively predict the
next character.

Source: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

Humans don’t start their thinking from scratch every second. As you read this essay, you understand each word based on your understanding of previous words. You don’t throw everything away and start thinking from scratch again. Your thoughts have persistence.

Traditional neural networks can’t do this, and it seems like a major shortcoming. For example, imagine you want to classify what kind of event is happening at every point in a movie. It’s unclear how a traditional neural network could use its reasoning about previous events in the film to inform later ones.

Recurrent neural networks address this issue. They are networks with loops in them, allowing information to persist.


[NEXT SECTION]
## 3. Deep Recurrent Networks

_note_
Source of following diagrams is:

http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/

[NEXT]
### Single neuron &mdash; one output

![rnn_diagram](images/rnn-perceptron.svg)

[NEXT]
### Neural network &mdash; multiple outputs

![rnn_diagram](images/rnn-feedforward.svg)

[NEXT]
### Deep Networks &mdash; many hidden layers

![deep_rnn_diagram](images/rnn-deepfeedforward.svg)

[NEXT]
### Simplified Visualisation

![rnn_compressed](images/rnn-compress.svg)

One node represents a full layer of neurons.

[NEXT]
![rnn_compress_expanded](images/rnn-compress-expanded.svg)

[NEXT]
### Recurrent Networks

![rnn_compressed](images/rnn-loopcompressed.svg)

Hidden layer's input includes the output of itself during the last run of the
network.

[NEXT]
### Unrolled Recurrent Network
Previous predictions help make the _next_ prediction.

Each prediction is a **time step**.

![rnn_unrolled](images/rnn-unrolled.svg)

[NEXT]
![rnn_unrolled_chars](images/rnn-unrolled-chars1.svg)

[NEXT]
![rnn_unrolled_chars](images/rnn-unrolled-chars2.svg)

[NEXT]
![rnn_unrolled_chars](images/rnn-unrolled-chars3.svg)

[NEXT]
![rnn_unrolled_chars](images/rnn-unrolled-chars4.svg)

[NEXT]
![rnn_unrolled_chars](images/rnn-unrolled-chars5.svg)

[NEXT]
![rnn_unrolled_chars](images/rnn-unrolled-chars6.svg)

[NEXT]
![rnn_unrolled_chars](images/rnn-unrolled-chars7.svg)

[NEXT]
![rnn_unrolled_chars](images/rnn-unrolled-chars8.svg)

[NEXT]
![rnn_unrolled_chars](images/rnn-unrolled-chars9.svg)

[NEXT]
### Problem: Long-Term Dependencies

![rnn_long_term_deps](images/rnn-long-term-deps.svg)

[NEXT]
### Cell States

Add **extra state** to each layer of the network.

Remembers inputs **far into the past**.

Transforms layer's original output into something that is **relevant** to
the current context.

_note_
This extra state stores information on inputs from much earlier time steps.

The state transforms the hidden layer's original output into something that is
relevant to the current context, given what's happened in the past.

[NEXT]
![rnn_compressed](images/rnn-stateloopcompressed.svg)

[NEXT]
![rnn_hiddenstate](images/rnn-hiddenstate.svg)

[NEXT]
Hidden layer output _and_ cell state is feed into next time step.

Gives network ability to handle long-term dependencies in sequences.

_note_
Feeding the output of a hidden layer _and_ its internal cell state back into
itself at the next time step allows the network to express long-term
dependencies in sequences.

For example, the network will be able to remember the subject at the start of
a paragraph of text at the very end of the paragraph, allowing it to generate
text that makes sense in context.

This works because the cell state is built to store past time steps in a
memory and CPU efficient way. So even though the cell state memory footprint is
small (e.g. 100-200 bytes), it can remember things quite far in the past.

[NEXT]
![rnn_long_term_deps](images/rnn-long-term-deps-solved.svg)

_note_

Note that there is still a limit to how far back networks with cell states can
remember. So we reduce the problems expressing long-term dependencies, but we
don't get rid of it entirely.

[NEXT]
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

_note_

Unfortunately, we don't have time in this talk to go into detail on how cell
states are represented and the different types.

So for this talk, we will treat the state as a "§box" and believe it
solves the long-term dependency problem.

Here's a link to a great article that explains the most commonly used cell
state technique in great detail.

[NEXT SECTION]
## 4. Training RNNs

[NEXT]
These recurrent networks are trained in the same way as regular network.

[NEXT]
**Backpropagation and gradient descent**.

![gradient_descent_small](images/gradient_descent_cropped.gif)

[NEXT]
![backprop_back_pass](images/backprop_backpass_3.svg)

[NEXT]
We need data to train the network.

[NEXT]
#### Gutenberg Datatset

Contains 34,000 English novels.

**https://www.gutenberg.org/**

[NEXT]
![novel_preprocessing](images/novel_preprocessing.svg)

[NEXT]
#### Common Training Methods

Run backpropagation after:

|                |                                |
| -------------- | ------------------------------ |
| **Stochastic** | one sequence                   |
| **Batch**      | all sequences                  |
| **Mini-Batch** | smaller batch of $b$ sequences |

[NEXT]
We'll use mini-batch.

[NEXT]
### Why?

|                                                   |                                                       |
| ------------------------------------------------- | ----------------------------------------------------- |
| <span style="color: red;">**Stochastic**</span>   | long time to converge on good weights                 |
| <span style="color: red;">**Batch**</span>        | consumes lots of memory, gets stuck on "okay" weights |
| <span style="color: green;">**Mini-Batch**</span> | quick to converge and memory efficient                |

[NEXT]
Iterate across all batches.

Run backpropagation after processing each batch.

![mini_batch_chars](images/mini-batch-chars.svg)

_note_
Split all the sequences into smaller batches of sequences.

Typically, batch size is between 30 and 100.


[NEXT SECTION]
## 5. Neural Nets in Python

[NEXT]
Building a neural network involves:

1. defining its architecture
2. learning the weight matrices for that architecture

_note_
1. e.g. number of layers, whether to have loops, what type of cell state to use
2. running an optimiser like gradient descent to train network on training dataset

[NEXT]
### Problem: complex graphs

![nn_computation_graph](images/nn_computation_graph.png)

_note_
Source: https://devblogs.nvidia.com/parallelforall/recursive-neural-networks-pytorch/

Here is a small section of the computation graph required to train a simple
recurrent network.

[NEXT]
![nn_backprop_algebra](images/nn_backprop_algebra.png)

_note_
Source: https://geekyisawesome.blogspot.co.uk/2016/06/the-backpropagation-algorithm-for.html

This is some of the algebra require for one step of backpropagaiton/training
for a single layer. And this is basic neural network with lno oops or cell states.

[NEXT]
### Python Neural Net Libraries

![icon_tensorflow](images/tensorflow_icon.svg)
![icon_keras](images/keras_icon.svg)
![icon_caffe](images/caffe_icon.svg)

![icon_pytorch](images/pytorch_icon.svg)
![icon_theano](images/theano_icon.svg)

_note_
Allows user to write symbolic mathematical expressions, then automatically generates their derivatives, saving the user from having to code gradients or backpropagation. These symbolic expressions are automatically compiled to CUDA code for a fast, on-the-GPU implementation.

Theano: The reference deep-learning library for Python with an API largely compatible with the popular NumPy library.

[NEXT]
![tensorflow](images/tensorflow.svg)

* Can build very complex networks quickly
* Easy to extend if required
* Built-in support for RNN memory cells
* Good visualisation tools


[NEXT SECTION]
## 6. Tensorflow

#### Quick Start

[NEXT]
### Goal

Build a computation graph that learns the weights of our network.

[NEXT]
### The Computation Graph

|                |                                                                                   |
| -------------  | --------------------------------------------------------------------------------- |
| `tf.Tensor`    | Unit of data. An _n_ dimensional array of numbers.        |
| `tf.Operation` | Unit of computation. Takes 0+ `tf.Tensor`s as inputs and outputs 0+ `tf.Tensor`s. |
| `tf.Graph`     | Collection of connected `tf.Tensor`s and `tf.Operation`s. |

Operations are nodes and tensors are edges.

[NEXT]
![tensorflow_graph_marked](images/tensorflow_graph_marked.png)

_note_
TODO: add tensorboard screenshot of the graph

[NEXT]
`tf.Operation`

Receives constants or `tf.Tensor`s as inputs.

Outputs `tf.Tensor`s.

[NEXT]
```python
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1)
print(node2)
```

Output:

```bash
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
```

[NEXT]
Many built-in `tf.Operation`s:

|            |                                                                    |
| ---------- | ------------------------------------------------------------------ |
| `constant` | outputs a constant tensor                                          |
| `reshape`  | reshapes an input tensor to a new tensor with different dimensions |
| `add`      | add values of two tensors                                          |
| `matmul`   | multiplys two matrices                                             |

[NEXT]
`tf.Session`

Used to execute a computation graph.

[NEXT]
```python
# Put an "addition" operation on the graph.
node3 = tf.add(node1, node2)

# Create a session and run the root node of the graph.
session = tf.Session()
print(session.run([node3]))
```

Output:

```bash
7.0
```

[NEXT]
`tf.Placeholder`

Defines a node whose value is not yet determined.

The value is filled in later, before the graph is executed.

These are used to define neural network inputs.

_note_
For us, the inputs will be those one-hot vector inputs that represent
characters which I showed you before.

[NEXT]
```python
# Build a graph to compute total cost of order.
# Input is a 2D vector containing: [price, quantity]
inputs = tf.placeholder(tf.float32, [2])
```

[NEXT]

```python
# Graph Node 1: inputs
inputs = tf.placeholder(tf.float32, [2])
# Graph Node 2: an internal operation
multiplied_inputs = tf.scalar_mul(3, inputs)
# Graph Node 3: final output
output_sum = tf.reduce_sum(multiplied_inputs)

# Run the graph.
session = tf.Session()
result = session.run(output_sum, feed_dict={inputs: [300, 10]})
print(f'₽{result}')
```

Output

```
₽30000
```

[NEXT]
![tensorflow_graph](images/tensorflow_graph.png)

[NEXT]

1. Define graph inputs: `tf.Placeholder`s
  - previous char that is used to predict the next char
2. Define graph operations: `tf.Operation`s
  - architecture of the network (hidden layers)
3. Use `tf.Session` to execute graph
  - call `run()`, passing in root node of graph


[NEXT SECTION]
## 7. Tensorflow

#### Building our Model

[NEXT]
![full_network](images/full_network_no_marks.svg)

[NEXT]
```python
SEQUENCE_LEN = 30
BATCH_SIZE = 200
ALPHABET_SIZE = 98
```

[NEXT]
![full_network_small](images/full_network_input.svg)

```python
# Dimensions: [ BATCH_SIZE, SEQUENCE_LEN ]
X = tf.placeholder(tf.uint8, [None, None], name='X')
```

[NEXT]
![full_network_small](images/full_network_onehot.svg)

```python
# Dimensions: [ BATCH_SIZE, SEQUENCE_LEN, ALPHABET_SIZE ]
Xo = tf.one_hot(X, ALPHABET_SIZE, 1.0, 0.0)
```

[NEXT]
![full_network_small](images/full_network_hidden.svg)

```python
HIDDEN_LAYER_SIZE = 512
NUM_HIDDEN_LAYERS = 3
```

_note_
Recap how the deep RNN cell layers work.

[NEXT]
```python
from tensorflow.contrib import rnn

# Cell State
# [ BATCH_SIZE, HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS]
H_in = tf.placeholder(
    tf.float32,
    [None, HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS],
    name='H_in')

# Create desired number of hidden layers that use the `GRUCell`
# for managing hidden state.
cells = [
    rnn.GRUCell(HIDDEN_LAYER_SIZE)
    for _ in range(NUM_HIDDEN_LAYERS)
]
multicell = rnn.MultiRNNCell(cells)
```

_note_
Point out that GRU cells is one of the most common methods for storing cell
states in hidden layers.

LSTM vs GRU difference:

From: https://www.quora.com/Are-GRU-Gated-Recurrent-Unit-a-special-case-of-LSTM

The other answer is already great. Just to add, GRU is related to LSTM as both
are utilizing different way if gating information to prevent vanishing gradient
problem.

GRU is relatively new, and from what I can see it's performance is on par with
LSTM, but computationally more efficient (less complex structure as pointed
out). So we are seeing it being used more and more.

[NEXT]
Wrap recurrent hidden layers in `tf.dynamic_rnn`.

Unrolls loops when computation graph is running.

Loops will be unrolled `SEQUENCE_LENGTH` times.

```python
Yr, H_out = tf.nn.dynamic_rnn(
    multicell,
    Xo,
    dtype=tf.float32,
    initial_state=H_in)

# Yr =    output of network. probability distribution of
#         next character.
# H_out = the altered hidden cell state after processing
#         last input.
```

_note_
The loops will be unrolled `SEQUENCE_LENGTH` times. You can think of this as us
copying all the hidden layer nodes for each unroll, creating a computation
graph that has 30 sets of hidden layers.

Note that `H_out` is the input hidden cell state that's been updated by the
last input. `H_out` is used as the next character's input (`H_in`).

[NEXT]
![rnn_unrolled](images/rnn-unrolled.svg)

[NEXT]
![full_network_small](images/full_network_softmax.svg)

```python
from tensorflow.contrib import layers

# [ BATCH_SIZE x SEQUENCE_LEN, HIDDEN_LAYER_SIZE ]
Yflat = tf.reshape(Yr, [-1, HIDDEN_LAYER_SIZE])
# [ BATCH_SIZE x SEQUENCE_LEN, ALPHABET_SIZE ]
Ylogits = layers.linear(Yflat, ALPHABET_SIZE)
# [ BATCH_SIZE x SEQUENCE_LEN, ALPHABET_SIZE ]
Yo = tf.nn.softmax(Ylogits, name='Yo')
```

_note_
Flatten the first two dimensions of the output:

[ BATCHSIZE, SEQLEN, ALPHASIZE ] => [ BATCHSIZE x SEQLEN, ALPHASIZE ]

Then apply softmax readout layer. The output of the softmax `Yo` is the
probability distribution

With this readout layer, the weights and biases are shared across unrolled time
steps. Doing this treats values coming from a single sequence time step (one
char) and values coming from a mini-batch run as the same thing.

[NEXT]
![full_network_small](images/full_network_output.svg)

```python
# [ BATCH_SIZE * SEQUENCE_LEN ]
Y = tf.argmax(Yo, 1)
# [ BATCH_SIZE, SEQUENCE_LEN ]
Y = tf.reshape(Y, [BATCH_SIZE, -1], name="Y")
```


[NEXT]
Remaining tasks:

* define our loss function
* decide what weight optimiser to use

[NEXT]
### Loss Function

Needs:

1. the **real** output of the network after each batch
2. the **expected** output (from our training data)

_note_
Used to compute a "loss" number that indicates how well the networking is
is predicting the next char.

[NEXT]
![full_network_loss](images/full_network_loss.svg)

[NEXT]

Input expected next chars into network:

```python
# [ BATCH_SIZE, SEQUENCE_LEN ]
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')

# [ BATCH_SIZE, SEQUENCE_LEN, ALPHABET_SIZE ]
Yo_ = tf.one_hot(Y_, ALPHABET_SIZE, 1.0, 0.0)

# [ BATCH_SIZE x SEQUENCE_LEN, ALPHABET_SIZE ]
Yflat_ = tf.reshape(Yo_, [-1, ALPHABET_SIZE])
```

[NEXT]
Defining the loss function:

```
# [ BATCH_SIZE * SEQUENCE_LEN ]
loss = tf.nn.softmax_cross_entropy_with_logits(
    logits=Ylogits,
    labels=Yflat_)

# [ BATCH_SIZE, SEQUENCE_LEN ]
loss = tf.reshape(loss, [BATCH_SIZE, -1])
```

_note_
We don't have time to cover the details of this loss function. All you need to
know for this talk is that is a commonly used loss function when predicting
discrete values like characters.

[NEXT]
Choose an optimiser.

Will adjust network weights to minimise the `loss`.

```
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
```

[NEXT SECTION]
## 8. Tensorflow

#### Training the Model

[NEXT]
### Epochs

We run mini-batch training on the network.

Train network on _all_ batches multiple times.

Each run across all batches is an **epoch**.

**More epochs = better weights = better accuracy.**

_note_
Of course, the downside of running loads of epochs is that it takes much
longer to train the network.

[NEXT]
```python
# Contains: [Training Data, Test Data, Epoch Number]
Batch = Tuple[np.matrix, np.matrix, int]

def rnn_minibatch_generator(
        data: List[int],
        batch_size: int,
        sequence_length: int,
        num_epochs: int) -> Generator[Batch, None, None]:

    for epoch in range(num_epochs):
        for batch in range(num_batches):
            # split data into batches, where each batch contains `b` sequences
            # of length `sequence_length`.
            training_data = ...
            test_data = ...
            yield training_data, c, epoch
```

_note_
Omit the details, just explain the underlying concept of splitting one big
large sequence into more sequences.

[NEXT]
```python
# Initialize the hidden cell states to 0 before running any steps.
input_state = np.zeros(
    [BATCH_SIZE, HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS])

# Create the session and initialize its variables to 0.
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
```

[NEXT]
Load dataset and construct mini-batch generator:

```python
char_integer_list = []
generator = rnn_minibatch_generator(
    char_integer_list,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    num_epochs=10)
```

[NEXT]
Run training step on all mini-batches for multiple epochs:

```python
# Initialise input state
step = 0
input_state = np.zeros([
  BATCH_SIZE, HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS
])

# Run training step loop
for batch_input, expected_batch_output, epoch in generator:
    graph_inputs = {
      X: batch_input,   Y_: expected_batch_output,
      Hin: input_state, batch_size: BATCH_SIZE
    }

    _, output, output_state = session.run(
        [train_step, Y, H],
        feed_dict=graph_inputs)

    # Loop state around for next recurrent run
    input_state = output_state
    step += BATCH_SIZE * SEQUENCE_LENGTH
```
<!-- .element class="small" -->

[NEXT]
### Demo

_note_
Briefly show real code and start running training. Run through one batch and
see how the loss is reduced.

[NEXT]
### Tensorboard

![tensorboard](images/tensorboard.png)

_note_
Explain what tensorboard is.

[NEXT]
### Final Results

[NEXT]
<!-- .slide: class="smallestquote" -->

**Epoch 1**

> TODO

Training time: TODO
<!-- .element class="medium" -->

(AWS p2.16x instance)
<!-- .element class="small" -->

[NEXT]
<!-- .slide: class="smallestquote" -->

**Epoch 5**

> TODO

Training time: TODO
<!-- .element class="medium" -->

(AWS p2.16x instance)
<!-- .element class="small" -->


[NEXT]
<!-- .slide: class="smallestquote" -->

**Epoch 10**

> TODO

Training time: TODO
<!-- .element class="medium" -->

(AWS p2.16x instance)
<!-- .element class="small" -->

[NEXT]
<!-- .slide: class="smallestquote" -->

**Epoch 100**

> Gradually drawing away from the rest, two of the combatants are striving; each devoting every nerve, every energy, to the overthrow of the other.
>
> But each attack is met by counter attack, each terrible swinging stroke by the crash of equally hard wood or the dull slap of tough hide shield opposed in parry.
>
> More men are down, about even numbers on each side, these two combatants strive on.

Training time: TODO
<!-- .element class="medium" -->

(AWS p2.16x instance)
<!-- .element class="small" -->


_note_
Show what text a pre-trained model can generate. Show multiple examples of what
it generates to bring the point home.

State how long it took to train that model.


[NEXT SECTION]
<!-- .slide: data-background="images/books_opened.jpg" class="background" -->
## Fin

[NEXT]
<!-- .slide: data-background="images/books_opened.jpg" class="background" -->
Machine learning often used for prediction.

Deep learning used to predict in domains where regular ML techniques cannot.

Deep recurrent neural networks good for predicting time-series data.

[NEXT]
<!-- .slide: data-background="images/books_opened.jpg" class="background" -->
Used RNNs to predict next chars in novels.

Trained RNN in Python using Tensorflow.

Trained model then used to generate real-looking text.

Significant computation required, but only a small amount of code.

_note_
Significant computation required, but only a small amount of code (thanks to
Tensorflow).

[NEXT]
<!-- .slide: data-background="images/books_opened.jpg" class="background" -->
### Slides
http://donaldwhyte.co.uk/deep-learning-with-rnns

[NEXT]
<!-- .slide: data-background="images/books_opened.jpg" class="background" -->
### Get In Touch

<table class="bio-table">
  <tr>
    <td>![small_portrait](images/donald.jpg)</td>
    <td>![small_portrait](images/alejandro.jpg)</td>
  </tr>
  <tr>
    <td>
      [don@donsoft.io](mailto:don@donsoft.io)<br />
      [@donald_whyte](http://twitter.com/donald_whyte)<br />
      <span class="github">https://github.com/DonaldWhyte</span>
    </td>
    <td>
      [a@e-x.io](mailto:a@e-x.io)<br />
      [@AxSaucedo](http://twitter.com/AxSaucedo)<br />
      <span class="github">https://github.com/axsauze</span>
    </td>
  </tr>
</table>

[NEXT]
<!-- .slide: data-background="images/books_opened.jpg" class="background" -->

### Sources

> [Martin Görner -- Tensorflow RNN Shakespeare](https://github.com/martin-gorner/tensorflow-rnn-shakespeare)

> [Composing Music with Recurrent Neural Networks](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/)
