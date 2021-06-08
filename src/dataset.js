
let fsPromises = require('fs').promises
let tfData = require('@tensorflow/tfjs-data');
let tfCore = require('@tensorflow/tfjs-core');
var lz4 = require('lz4');

let uniqueId = 0;

const FLOAT32TYPE = 2;
class Dataset extends tfData.Dataset {
    constructor(params={}){
        super();
        this._id = (uniqueId++);
        this.headerFile = (params.name||this._id)+'.ndlt'
        this.name = (params.name||this._id)+'.bin'
        this.struct = params.struct;
        
        this.compressLevel= 1;

        this.f =null;// file with dataset
        this.hf =null;// header file

        this._p = 0;//file position
        this._hp = 0;//header file position
        this._computeStrict(this.struct);

        this.fileOpen = Promise.all([fsPromises.open(this.headerFile, 'a+'), fsPromises.open(this.name, 'a+')]).then(([hf,f])=>{
            this.f = f;
            this.hf = hf;
            let magicSymbol = Buffer.from('ndlt')
            let structSerialized = Buffer.from(JSON.stringify(this.struct));

            let header = Buffer.concat([magicSymbol,Buffer.from(new Uint32Array([structSerialized.length])),structSerialized, Buffer.from([this.compressLevel,0,0,0,0])]);

            this._hp+=header.length;

            //write header and clear file
            return Promise.all([this.hf.write(header), this.hf.truncate(this._hp),this.f.truncate(0) ])
        });
        this.list = [];
        
        
       
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
        this.compressBuf = Buffer.alloc(lz4.encodeBound(len*4));
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

        let hp = this._hp;
        let dataInfo = new BigUint64Array([BigInt(p), BigInt(length), BigInt(FLOAT32TYPE), 0n]);
        
        this.list.push(dataInfo);

        this._hp+=dataInfo.byteLength;
        this._p+=length;

        await this.fileOpen;

        let writing = this.f.write(Buffer.from(cloned.buffer, cloned.byteOffset, length), 0, length, p);
        await writing;


        let writingHeader = this.hf.write(Buffer.from(dataInfo.buffer, dataInfo.byteOffset, dataInfo.byteLength), 0, dataInfo.byteLength, hp);
        await writingHeader;
    }
    async get(dataInfo){
        


        let out = Buffer.from(this.buf.buffer, this.buf.byteOffset, this.buf.byteLength);

        await this.f.read(this.decompressBuf, 0, Number(dataInfo[1]), Number(dataInfo[0]));
        var uncompressedSize = lz4.decodeBlock(this.decompressBuf.subarray(0,Number(dataInfo[1])), out)

        if(uncompressedSize !== this.buf.byteLength){
            console.log(Number(dataInfo[0]), Number(dataInfo[1]));
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
                out[key] = await this.datasets[key].get(this.datasets[key].list[i]);
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

        for(let k=0;k!== this.keys.length;k++){
            let key = this.keys[k];
            await this.datasets[key].fileOpen;
            
        }
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