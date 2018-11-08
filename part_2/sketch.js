console.log("welcome to machine learning session");


// build the model
const model = tf.sequential();

// add a hidden Layer
const hidden = tf.layers.dense({
    units: 4,
    inputShape: [2],
    activation: 'sigmoid'
});
model.add(hidden);

// output layer
const output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
});
model.add(output);

// compile the model
const sgdOpt = tf.train.sgd(0.1);
model.compile({
    optimizer: sgdOpt,
    loss: tf.losses.meanSquaredError
});


const xs = tf.tensor2d([
    [0, 0],
    [0.5, 0.5],
    [1, 1]
]);

const ys = tf.tensor2d([
    [0],
    [0.5],
    [1],
]);

//train the model
async function trainModel() {
    for (let index = 0; index < 1000; index++) {
        const config = {            
            shuffle: true,
            epochs: 10
        }
        const h = await model.fit(xs, ys, config);
        console.log("Loss after Epoch " + index + " : " + h.history.loss[0]);
    }
}

trainModel().then(() => {
    console.log("train complete")
    // predict the output
    let output1 = model.predict(xs);
    output1.print();
});

