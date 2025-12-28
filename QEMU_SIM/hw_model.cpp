/*
    아래 코드들은 추후 하드웨어 구현을 위해, 모든 경우를 다 표현하였으므로
    사용되지 않는 {} 복합표현문을 일단 모두 기입하였음.
*/

#include "hw_model.hpp"

Icosahedron::Icosahedron(){
    bool triangleType;
    for (unsigned char faceIdx = 0; faceIdx < NUM_OF_FACE; faceIdx++){       
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

void Icosahedron::setFaceAttr(){
    for (unsigned char faceIdx = 0; faceIdx < NUM_OF_FACE; faceIdx++){
        /*
            1. Get my position
            2. Set neighborhood faces to use my position

            +++ Refactoring idea
              - reduce branch??
        */
        
        char myFaceId = faceIdx;
        char ele, left, right;
        
        /*  GET MY POSITION
            - Odd/even?
            - Under/upper 10?
        */
        if (myFaceId % 2 == EVEN){
            // Center
            ele = myFaceId + 1;
            if (myFaceId >= 10){
                left = myFaceId-10;

                if (myFaceId != 18) right = myFaceId+2;
                else if (myFaceId == 18) right = 0;
                else {} // Nothing
            }
            else if (myFaceId < 10){
                if (myFaceId != 0) left = myFaceId-8;
                else if (myFaceId == 0) left = 18;
                else // Nothing

                right = myFaceId+10;
            }
            else {} // Nothing
        }
        else if (myFaceId % 2 == ODD){
            // Upper or Under
            ele = myFaceId - 1;
            left = myFaceId - 2;
            right = myFaceId + 2;

            // 10 up/down?
            if (myFaceId >= 10){
                if (left < 10) left = 19;
                else if (left >= 10) {} // Nothing
                else {} // Nothing

                if (right >= 20) right = 11;
                else if (right < 20) {} // Nothing
                else {} // Nothing
            }
            else if (myFaceId < 10){
                if (left < 0) left = 9;
                else if (left >= 0) {} // Nothing
                else {} // Nothing

                if (right >= 10) right = 1;
                else if (right < 10) {} // Nothing
                else {} // Nothing
            }
            else {} // Nothing
        }
        else {} // Nothing
    }
}

IcosaFaceTri* Icosahedron::getStartFace(
        unsigned char azimuth, char elevation) {
    // Get start Position
    char tri;
    char tri_start = (azimuth / 72);
    char azi_norm = (azimuth / 36); 

    enum odd_even azi_odd_even = (azi_norm / 18 == 0)? EVEN : ODD;
    
    enum ele_area ea = (elevation < -30)? BOTTOM :
                        (elevation < 0)? MIDDLE_BOTTOM :
                        (elevation < 30)? MIDDLE_TOP :
                        TOP;
    
    switch(ea){
        case BOTTOM:
            tri = tri_start + 1;
        break;
        case MIDDLE_BOTTOM:
            tri = tri_start;
        break;
        case MIDDLE_TOP:
            if (azi_odd_even == EVEN){
                tri = tri_start - 10;
            }
            else if (azi_odd_even == ODD){
                tri = tri_start + 10;
            }
        break;
        case TOP:
           tri = tri_start + 11;
        break;
    }

    // tri determine when out of range
    if (tri >= 20){ tri -= 20; }
    else if (tri < 20) {} // Nothing
    else {} // Nothing

    if (tri < 0){ tri += 20; }
    else if (tri >= 0) {} // Nothing
    else {} // Nothing

    return faces[tri];
}

IcosaFaceTri** getPovFaces(
            unsigned char *num_of_tris, IcosaFaceTri* startFace,
            unsigned char azimuth, char elevation, 
            unsigned char pov_x, unsigned char pov_y, unsigned char roll
        ){
    // Get POV triangles
    IcosaFaceTri** tri_pov;
    unsigned char firstAzi; char firstEle;
        // Get center tri azimuth, elevation
    

    unsigned char triAzi; char triEle;
    
    // - todo
}

IcosaFaceTri::IcosaFaceTri(unsigned char faceIdx, bool triangleType){
    faceId = faceIdx;
    triType = triangleType;
}

void IcosaFaceTri::setCloserFaces(IcosaFaceTri* ele, IcosaFaceTri* left, IcosaFaceTri* right){
    elevateFace = ele; leftFace = left; rightFace = right;
}

unsigned char IcosaFaceTri::getFaceId() { return faceId; }
bool IcosaFaceTri::getTriType() { return triType; }
IcosaFaceTri* IcosaFaceTri::getElevateFace() { return elevateFace; }
IcosaFaceTri* IcosaFaceTri::getLeftFace() { return leftFace; }
IcosaFaceTri* IcosaFaceTri::getRightFace() { return rightFace; }

IcosaFaceTri* IcosaFaceTri::getFaceFromAngle(unsigned short angle) {
    // start direction is noontime and clockwise
    if (!triType){ // Shape: "/\"
        if (angle >= ANGLE_0 && angle < ANGLE_120) return rightFace;
        else if (angle >= ANGLE_120 && angle < ANGLE_240) return elevateFace;
        else if (angle >= ANGLE_240 && angle < ANGLE_0) return leftFace;
        else {} // Nothing
    }
    else if (triType){ // Shape: "\/"
        if (angle >= ANGLE_60 && angle < ANGLE_180) return rightFace;
        else if (angle >= ANGLE_180 && angle < ANGLE_300) return leftFace;
        else if (angle >= ANGLE_300 && angle < ANGLE_MAX) return elevateFace;
        else if (angle >= ANGLE_0 && angle < ANGLE_60) return elevateFace;
        else {} // Nothing
    }
    else {} // Nothing
}
