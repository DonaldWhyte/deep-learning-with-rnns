# Deep Learning with Recurrent Neural Networks

### [View Presentation Slides](http://donaldwhyte.github.io/deep-learning-with-rnns/)

Presentation briefly introducing deep learning and how to apply a specific subset of deep learning, recurrent neural networks, to solve real world problems.

Topics covered:

* basics of supervised learning and classification
* perceptrons
* neural networks
* recurrent neural networks
* cell states (LSTM and GRU networks)
* building computation graphs in Tensorflow
* building, training and running RNNs in Tensorflow

## Workshop

There is a workshop that accompanies this talk. The workshop involves building the RNN architecture discussed in the talk to generate Shakespearean text.

### [View Workshop Slides](http://donaldwhyte.github.io/deep-learning-with-rnns/workshop.html)

### [Workshop Code Solutions and Data](workshop/)

## Running Presentation

You can also run the presentation on a local web server. Clone this repository and run the presentation like so:

```
npm install
grunt serve
```

The presentation can now be accessed on `localhost:8080`. Note that this web application is configured to bind to hostname `0.0.0.0`, which means that once the Grunt server is running, it will be accessible from external hosts as well (using the current host's public IP address).

## Acknowledgements

Huge credit to **[Martin Görner](https://github.com/martin-gorner/)** for his awesome [Tensorflow RNN Shakespere](https://github.com/martin-gorner/tensorflow-rnn-shakespeare) talk, which inspired a lot of the code in this talk/workshop.
