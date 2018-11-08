console.log("welcome to machine learning session");

function setup() {
    noCanvas();    
    //frameRate(1);
    //tense.print();
    //console.log(tense.get(29));
    //console.log(a.dataSync());
    
}

function draw() {
    const values = [];

    for (let index = 0; index < 15; index++) {
        values[index] = random(0,100);
    }    
    const shape = [5 , 3];
        
    tf.tidy(()=> {
        const a = tf.tensor2d(values, shape, 'int32');
        const b = tf.tensor2d(values, shape, 'int32');
        const bb = b.transpose();
        const c = a.matMul(bb);
    })


    console.log(tf.memory().numTensors);
   // console.log('hello');
}
