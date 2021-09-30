let tf = require('@tensorflow/tfjs')
let { Dataset } = require('./../src/dataset')

let inpSize = 190000
async function trainModel() {

  {
    let time = new Date();
    const xArray = [];
    const yArray = [];
    for (let i = 0; i !== 300; i++) {
      //xArray.forEach((el)=>{
      //  xDataset.push(el)
      //})
      if (i % 256 === 0) {
        console.log('i', i);
      }
      let data = new Float32Array(inpSize)
      // for(let p=0;p!==inpSize<<1;p++){
      //   data[p]= Math.random()
      // }
      xArray.push(data)
      yArray.push(new Float32Array(1))

    }

    const xDataset = tf.data.array(xArray.map(v => tf.tensor(v, [1, inpSize])));
    const yDataset = tf.data.array(yArray.map(v => tf.tensor(v, [1, 1])));

    console.log('time', new Date() - time);
    const xyDataset = tf.data.zip({ xs: xDataset, ys: yDataset })//zip({xs: xDataset, ys: yDataset})
    // .batch(4)
    // .shuffle(4);
    const model = tf.sequential({
      layers: [tf.layers.dense({ units: 1, inputShape: [inpSize] })]
    });
    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

    await model.fitDataset(xyDataset, {
      epochs: 1, batchesPerEpoch: 256,
      callbacks: {
        onEpochEnd: (epoch, logs) => console.log('onEnd', epoch, logs.loss)
      }
    });
    console.log('time', new Date() - time);
  }
  {
    const xDataset = new Dataset({ shape: [inpSize] })
    const yDataset = new Dataset({ shape: [1] })
    let time = new Date();
    for (let i = 0; i !== 300; i++) {
      //xArray.forEach((el)=>{
      //  xDataset.push(el)
      //})
      if (i % 256 === 0) {
        console.log('i', i);
      }
      let data = new Float32Array(inpSize)
      // for(let p=0;p!==inpSize<<1;p++){
      //   data[p]= Math.random()
      // }
      xDataset.send(data)
      yDataset.send(new Float32Array(1))
    }



    console.log('time', new Date() - time);
    const xyDataset = tf.data.zip({ xs: xDataset.generator(), ys: yDataset.generator() })
    // .batch(4)
    // .shuffle(4);


    const model = tf.sequential({
      layers: [tf.layers.dense({ units: 1, inputShape: [inpSize] })]
    });

    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
    await model.fitDataset(xyDataset, {
      epochs: 1,
      batchesPerEpoch: 256,
      callbacks: { onEpochEnd: (epoch, logs) => { console.log('onEnd', epoch, logs.loss) } }
    });
    console.log('time', new Date() - time);

    xDataset.destroy();
    yDataset.destroy();

  }
  // setInterval(()=>{},1000)
}
trainModel().catch(err => {
  console.log(err);
  process.exit(1)
});