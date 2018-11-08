console.log("welcome to machine learning session");


// fetch JSON file
async function fetchJSON(url) {
    return new Promise((resolve, reject) => {
        fetch(url)
            .then(
                function (response) {
                    if (response.status !== 200) {
                        console.log('Looks like there was a problem. Status Code: ' +
                            response.status);
                        return;
                    }

                    // Examine the text in the response
                    response.json().then(function (data) {
                        resolve(data);
                    });
                }
            )
            .catch(function (err) {
                reject("there is an array");
                console.log('Fetch Error :-S', err);
            });
    });
}

// train the model
async function trainModel(model, input, output) {
    const config = {
        shuffle: true,
        epochs: 10
    }

    for (let index = 0; index <100; index++) {        
        const h = await model.fit(input, output, config);
        console.log("Loss after Epoch " + index + " : " + h.history.loss[0]);
        
    }

}

function prediction(model, testData) {
    const output = model.predict(testData);
    output.print();
}

async function run() {
    const tranningUrl = "iris.json";
    const testingUrl = "iris-testing.json";

    const tData = await get(tranningUrl);
    const testData = await get(testingUrl);    
    const model = buildModel();

    const trainingData = tf.tensor2d(tData.map(item => [
        item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
    ]))
    const outputData = tf.tensor2d(tData.map(item => [
        item.species === "setosa" ? 1 : 0,
        item.species === "virginica" ? 1 : 0,
        item.species === "versicolor" ? 1 : 0,
    ]))

    console.log(outputData.print());
    const testingData = tf.tensor2d(testData.map(item => [
        item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
    ]))
    trainModel(model, trainingData, outputData).then(() => {
        console.log("train complete")
        // predict the output
        const output = model.predict(testingData);
        output.print();
    });
}

run();

function buildModel() {
    const model = tf.sequential();
    // add a hidden Layer
    const hidden1 = tf.layers.dense({
        units: 5,
        inputShape: [4],
        activation: 'sigmoid'
    });
    model.add(hidden1);

    const hidden2 = tf.layers.dense({
        inputShape: [5],
        activation: "sigmoid",
        units: 3,
    });
    model.add(hidden2);

    // output layer
    const output = tf.layers.dense({
        units: 3,
        activation: 'sigmoid'
    });
    model.add(output);

    model.compile({
        optimizer: tf.train.adam(.06),
        loss: tf.losses.meanSquaredError
    });
    return model;
}

async function get(url) {
    const data = await fetchJSON(url);
    return data;
}