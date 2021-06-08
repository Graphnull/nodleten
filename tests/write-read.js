let tf = require('@tensorflow/tfjs')
let data = require('./data')
let {Dataset, zip} = require('./../src/dataset')
let fs = require('fs')


function Float32Concat(first, second)
{
    var firstLength = first.length,
        result = new Float32Array(firstLength + second.length);

    result.set(first);
    result.set(second, firstLength);

    return result;
}



async function test() {

    let dataset = new Dataset({shape:[6+1000000], name:'test'});


    for(let i=0;i!==1000;i++){
        let addData = new Float32Array(1000000);
        for(let j=0;j!==10000;j++){
            addData[j]= Math.random();
        }
        let data = Float32Concat(new Float32Array([i,0,1,2,3,i]), addData)
        dataset.push(data)
        await new Promise(res=>setTimeout(res,1))
    }

    let i=0;
    await dataset.forEachAsync(async(tensor)=>{
        let j = i++;
        let data = tensor.dataSync();
        if(data[0]!=j || data[1]!=0 || data[2]!=1 || data[3]!=2 || data[4]!=3 || data[5]!=j){
            throw new Error(''+j+' not equal'+data.slice(0,6))

        }
    })

    await dataset.writeHeaderFile();
    console.log(process.memoryUsage());
    if(process.memoryUsage().rss>200*1000*1000){
        throw new Error(`data not cached (${process.memoryUsage().rss/1000/1000}>${200})`)

    }
    let files = fs.readdirSync('./')
    
    if(!files.find(f=>f==='test.bin')||!files.find(f=>f==='test.ndlt')){
        throw new Error('test file not found')
    }
    if(!(fs.statSync('./test.bin').size>0)){
        throw new Error('test.bin file small')
    }
    if(!(fs.statSync('./test.ndlt').size>0)){
        throw new Error('test.ndlt file small')
    }
    console.log('success')
    dataset.destroy();

}
test().catch(err=>{
    console.error(err);
    process.exit(1)
});