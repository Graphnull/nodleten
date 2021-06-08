
let fsPromises = require('fs').promises
let tfData = require('@tensorflow/tfjs-data');
let tfCore = require('@tensorflow/tfjs-core');
var lz4 = require('lz4');
const {
    Worker, isMainThread, parentPort, workerData
} = require('worker_threads');

let uniqueId = 0;

const FLOAT32TYPE = 2;
if(isMainThread){
 
let uniqueCommandId =0;
class Dataset extends tfData.Dataset {
    constructor(params={}){
        super();
        let workerData = params;
        workerData.name = params.name||(uniqueId++);
        workerData.headerFile = (this.name)+'.ndlt'
        workerData.dataFile = (this.name)+'.bin'
        workerData.struct = params.struct;
        
        workerData.compressLevel= typeof params.compressLevel ==='number'?params.compressLevel:1;

        this.worker = new Worker(__filename, {
            workerData: workerData
          });
        this.worker.dataI =0;
        this.list=[];
    }
    push(objs){
        this.list.push(this.worker.dataI++);
        let clone = objs.slice(0);
        this.worker.postMessage({op:'push',data:clone},[clone.buffer])
    }
    send(objs){
        this.list.push(this.worker.dataI++);
        this.worker.postMessage({op:'push',data:objs},[objs.buffer])
    }
    async get(i){

        let index = this.list[i];
        let commandId = uniqueCommandId++;
        this.worker.postMessage({index,id:commandId,op:'get'})
        let result = await new Promise((res)=>{
            let onMessage = (message)=>{
                if(message.id === commandId){
                    this.worker.off('message', onMessage)
                    res(message.data)
                }
            }
            this.worker.on('message', onMessage)

        })
        return result;
    }
}
class Zip {
    constructor(datasets){
        
        this.datasets = datasets;
        this.firstDataset = Object.values(datasets)[0]
        this.keys = Object.keys(datasets)
    }
    batch (){throw new Error('not implemented')}
    concatenate (){throw new Error('not implemented')}
    filter (){throw new Error('not implemented')}
    forEachAsync (){throw new Error('not implemented')}
    map (){throw new Error('not implemented')}
    mapAsync (){throw new Error('not implemented')}
    prefetch (){throw new Error('not implemented')}
    repeat (){throw new Error('not implemented')}
    shuffle (){throw new Error('not implemented')}
    skip (){throw new Error('not implemented')}
    take (){throw new Error('not implemented')}
    toArray (){throw new Error('not implemented')}
    async forEachAsync(func){

        let list = this.firstDataset.list;
        for(let i=0;i!== list.length;i++){
            let out = {}

            for(let k=0;k!== this.keys.length;k++){
                let key = this.keys[k];
                out[key] = await this.datasets[key].get(i);
            }
            
            tfCore.tidy(()=>{
                for(let k=0;k!== this.keys.length;k++){
                    let key = this.keys[k];
                    out[key] = tfCore.tensor(out[key],[1, out[key].length]);
                }
                func(out);
            })
        }
        

    }
    async iterator(){
        let i=0;
        let count = this.firstDataset.list.length;

        return {
            next: async ()=>{
                if(i%256===0){
                    console.log('i',i);
                }
                if(count===i){
                    return {value: null,done:true};
                }
                i++;
                let out = {}

                for(let k=0;k!== this.keys.length;k++){
                    let key = this.keys[k];
                    out[key] = await this.datasets[key].get(this.datasets[key].list[i]);
                    
                }
                
                for(let k=0;k!== this.keys.length;k++){
                    let key = this.keys[k];
                    out[key] = tfCore.tensor(out[key],[1, out[key].length]);
                }
                return {value:out,done:false};
            }
        }
    }
}
function zip(objects){

    return new Zip(objects);
}


module.exports.zip = zip;
module.exports.Dataset = Dataset;
}else{
//const {performance} = require('perf_hooks');
    const inspector = require('inspector');
    
const session = new inspector.Session();
session.connect()

session.post('Profiler.enable', () => {
    session.post('Profiler.start', () => {})
})
class Dataset {
    constructor(params={}){
        this.name = params.name||(uniqueId++);
        this.dataFile = (this.name)+'.bin'
        this.struct = params.struct;
        
        this.compressLevel= typeof params.compressLevel ==='number'?params.compressLevel:1;

        this.f =null;// file with dataset

        this._p = 0;//file position
        this._computeStrict(this.struct);

        this.fileOpen = fsPromises.open(this.dataFile, 'a+').then((f)=>{
            this.f = f;
            //clear file
            return this.f.truncate(0)
        });

        this.list = [];
        
        this.queue={};
       
        this.writeQueue=[];
    }
    writeHeaderFile (){

        //TODO
        let headerFile = (this.name)+'.ndlt'
        let magicSymbol = Buffer.from('ndlt')
        let structSerialized = Buffer.from(JSON.stringify(this.struct));
        let header = Buffer.concat([magicSymbol,Buffer.from(new Uint32Array([structSerialized.length])),structSerialized, Buffer.from([this.compressLevel,0,0,0,0])]);

        this._hp+=header.length;
    }
    _computeStrict(){
        if(!Array.isArray(this.struct)){
            throw new Error('Struct field not array');
        }

        let len = 0;

        this.struct.forEach(shape=>{
            if(!Array.isArray(shape)){
                throw new Error(JSON.stringify(shape)+' shape is not number array');
            }
            shape.forEach(dim=>{
                if(typeof dim !=='number'){
                    throw new Error(JSON.stringify(shape)+' shape is not number array');
                }
                len+=dim|0
            })
        })
        this.buf = new Float32Array(len);
        this.compressBuf = Buffer.alloc((len*4)*2);//Buffer.alloc(lz4.encodeBound(len*4));
        this.decompressBuf = Buffer.alloc(this.compressBuf.length);
    }

    batch (){throw new Error('not implemented')}
    concatenate (){throw new Error('not implemented')}
    filter (){throw new Error('not implemented')}
    forEachAsync (){throw new Error('not implemented')}
    map (){throw new Error('not implemented')}
    mapAsync (){throw new Error('not implemented')}
    prefetch (){throw new Error('not implemented')}
    repeat (){throw new Error('not implemented')}
    shuffle (){throw new Error('not implemented')}
    skip (){throw new Error('not implemented')}
    take (){throw new Error('not implemented')}
    toArray (){throw new Error('not implemented')}
    async push(objs){
        if(!objs){
            throw new Error('Data not found')
        }

        // let offset = 0;
        // Object.keys(objs).forEach(key=>{
        //     let data = objs[key];
        //     if(!Array.isArray(data)){
        //         throw new Error(JSON.stringify(data)+key+' data is not array');
        //     }

        //     data.forEach(arr=>{
        //         if(!(data instanceof Float32Array)){
        //             throw new Error(JSON.stringify(arr)+key+' data is not Float32Array');
        //         }
        //     })
        //     this.buf.set(arr, offset);
        //     offset+=arr.length;
        // })
 
        let offset = 0;
        let data = objs;

      
        
        this.buf.set(data, offset);

        let time = new Date();
        var compressedSize = lz4.encodeBlock(Buffer.from(this.buf.buffer, this.buf.byteOffset, this.buf.byteLength), this.compressBuf)

        if(compressedSize>this.compressBuf.length){
            throw new Error(`compressBuf.length (${this.compressBuf.length})<compressedSize (${compressedSize})`)
        }

        let cloned = this.compressBuf.slice(0)
        offset+=compressedSize;

        let p = this._p;
        let length = compressedSize
        this._p+=length;

        let dataInfo = new BigUint64Array([BigInt(p), BigInt(length), BigInt(FLOAT32TYPE), 0n]);
        
        this.list.push(dataInfo);

        this._p+=length;
        let writeData = Buffer.from(cloned.buffer, cloned.byteOffset, length);
        this.queue[p]={writeData}

        // let writeQueueLength = 0;
        // this.writeQueue.forEach((data)=>{
        //     writeQueueLength+=data;
        // })
        // if((writeQueueLength+writeData.length)>(1024*1024*4)){
            await this.fileOpen;
            let writing = this.f.write(writeData, 0, length, p);
            //let writing = this.f.write(Buffer.concat(this.writeQueue.concat([writeData])), 0, length, p);
            await writing;
            this.queue=[];
        //}
      


    }
    async get(dataInfo){
        

        let p = Number(dataInfo[0]);
        let len = Number(dataInfo[1]);
        let out = Buffer.from(this.buf.buffer, this.buf.byteOffset, this.buf.byteLength);
        var uncompressedSize = 0;
        if(this.queue[p]){
            let writeData = this.queue[p].writeData;
            uncompressedSize = lz4.decodeBlock(writeData, out)
        }else{
            let copy = this.decompressBuf.slice(0);
            await this.f.read(copy, 0, len, p);
            uncompressedSize = lz4.decodeBlock(copy.subarray(0,len), out)
        }

        if(uncompressedSize !== this.buf.byteLength){
            console.log(p, len,this.decompressBuf.length );
            throw new Error(`uncompressedSize (${uncompressedSize}) !== this.buf.byteLength (${this.buf.byteLength})`)
        }

        return this.buf
    }
    async forEachAsync(func){
        await this.fileOpen;

        for(let i=0;i!== this.list.length;i++){

            let out = await this.get(this.list[i]);
            
            tfCore.tidy(()=>{
                func(tfCore.tensor(out));
            })
        }
        

    }
    async iterator(){

    }
    _writeFile(){
        this.f.write(header);
    }
}

let dataset = new Dataset(workerData);


// setInterval(()=>{
//     let lenght = 0;
//     Object.values(dataset.queue).forEach(q=>{
//         lenght+=q.writeData.length
//     })
//     console.log('stats', (lenght/1024/1024).toFixed(2));
// },100)
let time = null;
let startMark = null;
let endMark = null;
parentPort.on('message', (message) => {
    switch(message.op){
        case('get'):{
            if(time){
                //endMark = performance.mark('first')
                async function mark1(){
                    await new Promise((res)=>setTimeout(res,100))
                    session.post('Profiler.stop', (err, { profile }) => {
                    
                    // Write profile to disk, upload, etc.
                    if (!err) {
                        fsPromises.writeFile('./profile2.cpuprofile', JSON.stringify(profile));
                    }
                });
                }
                mark1()
            
                console.log('first', new Date()-time);
                time = null;
            }
            let dataInfo = dataset.list[message.index];
            if(!dataInfo){
                console.log('ddddd',dataset.list.length);
                throw new Error(`data in ${message.index} index not found`)
            }
            
            dataset.get(dataInfo).then((data)=>{
                parentPort.postMessage({id:message.id,data});
            })
        break;
        }
        case('push'):{// push
            dataset.push(message.data);
            time = new Date();
            //startMark = performance.mark('end1')
            //parentPort.postMessage(message);
            break;
        }
    }
  });
}
