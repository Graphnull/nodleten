let tf = require('@tensorflow/tfjs')
let data = require('./data')
let {Dataset, zip} = require('./../src/dataset')

let inpSize = 190000
async function trainModel() {

{
  let time = new Date();
  const xArray = [];
 const yArray = [];
 for(let i=0;i!==6000;i++){
  //xArray.forEach((el)=>{
  //  xDataset.push(el)
  //})
  if(i%256===0){
    console.log('i',i);
  }
  xArray.push(new Float32Array(inpSize))
  yArray.push(new Float32Array(1))

  }

  const xDataset = tf.data.array(xArray.map(v=>tf.tensor(v,[1, inpSize]) ));
  const yDataset = tf.data.array(yArray.map(v=>tf.tensor(v,[1, 1])));

  console.log('time', new Date()-time);
  const xyDataset = tf.data.zip({xs: xDataset, ys: yDataset})//zip({xs: xDataset, ys: yDataset})
        // .batch(4)
        // .shuffle(4);
  const model = tf.sequential({
      layers: [tf.layers.dense({units: 1, inputShape: [inpSize]})]
  });
  model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
console.log(22);
  await model.fitDataset(xyDataset, {
      epochs: 1,batchesPerEpoch:1024,
      callbacks: {
        onEpochEnd: (epoch, logs) => console.log('onEnd',epoch, logs.loss)
      }
  });
  console.log('time', new Date()-time);
}
{
  let time = new Date();
  const xDataset = new Dataset({struct:[[inpSize]]})
  const yDataset = new Dataset({struct:[[1]]})
  for(let i=0;i!==6000;i++){
  //xArray.forEach((el)=>{
  //  xDataset.push(el)
  //})
  if(i%256===0){
    console.log('i',i);
  }
  await xDataset.push(new Float32Array(inpSize))
  yDataset.push(new Float32Array(1))
  }



  console.log('time', new Date()-time);
  const xyDataset = zip({xs: xDataset, ys: yDataset})
      // .batch(4)
      // .shuffle(4);


  const model = tf.sequential({
      layers: [tf.layers.dense({units: 1, inputShape: [inpSize]})]
  });

  model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
  await model.fitDataset(xyDataset, {
      epochs: 1,
      batchesPerEpoch:1024,
      callbacks: {onEpochEnd: (epoch, logs) => {console.log( 'onEnd',epoch, logs.loss)}}
  });
  console.log('time', new Date()-time);

}
 // setInterval(()=>{},1000)
  }
  trainModel();