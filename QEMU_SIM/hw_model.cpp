#include "hw_model.hpp"

Icosahedron::Icosahedron(){
    bool triangleType;
    for (unsigned char faceIdx = 0; faceIdx < NUM_OF_FACE; faceIdx++){
        /*  0 <= n <= 4
                 /\    /\    /\    /\    /\
                /11\  /13\  /15\  /17\  /19\     (10+2n+1) false
              /\----/\----/\----/\----/\----/
             / 0\10/ 2\12/ 4\14/ 6\16/ 8\18/     (2n or 10+2n) true / false
            /----\/----\/----\/----\/----\
             \ 1/  \ 3/  \ 5/  \ 7/  \ 9/        (2n+1) false
              \/    \/    \/    \/    \/
        */        
        triangleType = (faceIdx < 10)? 
                            ( (faceIdx % 2 == 0)? false : true ) : 
                            ( (faceIdx % 2 == 0)? true : false );
        faces[faceIdx] = new IcosaFaceTri(faceIdx, triangleType);
    }
}

Icosahedron::~Icosahedron(){
    for (unsigned char faceIdx = 0; faceIdx < NUM_OF_FACE; faceIdx++){
        delete faces[faceIdx];
    }
}

IcosaFaceTri* Icosahedron::getFaceAngle(unsigned char normAngle){
    
    return faces[];
}