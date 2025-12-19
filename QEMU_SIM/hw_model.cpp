#include "hw_model.hpp"

Icosahedron::Icosahedron(){
    bool triangleType;
    for (unsigned char faceIdx = 0; faceIdx < NUM_OF_FACE; faceIdx++){
        /*  0 <= n <= 4
                 /\    /\    /\    /\    /\
                /11\  /13\  /15\  /17\  /19\     (10+2n+1) false
              /\----/\----/\----/\----/\----/
             / 0\10/ 2\12/ 4\14/ 6\16/ 8\18/     (2n or 10+2n) true / false
            /----\/----\/----\/----\/----\/
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

IcosaFaceTri** Icosahedron::getFaceAngle(
    unsigned char *num_of_tris,
    unsigned char azimuth, char elevation, 
    unsigned char pov_x, unsigned char pov_y, unsigned char roll
){
    // Get start Position
    char tri;
    char tri_start = (azimuth / 72);
    char azi_norm = (azimuth / 36); 
    
    ODD_EVEN oe;

    bool azi_odd_even = (azi_norm / 18 == 0)? oe.EVEN : oe.ODD;
    
    ELE_AREA ele_area = (elevation < -30)? BOTTOM :
                        (elevation < 0)? MIDDLE_BOTTOM :
                        (elevation < 30)? MIDDLE_TOP :
                        TOP;
    
    switch(ele_area){
        case BOTTOM:
            tri = tri_start + 1;
        break;
        case MIDDLE_BOTTOM:
            tri = tri_start;
        break;
        case MIDDLE_TOP:
            if (azi_odd_even == oe.EVEN){
                tri = tri_start - 10;
            }
            else if (azi_odd_even == oe.ODD){
                tri = tri_start + 10;
            }
        break;
        case TOP:
           tri = tri_start + 11;
        break;
    }

    // Get POV triangles
    IcosaFaceTri** tri_pov;
}