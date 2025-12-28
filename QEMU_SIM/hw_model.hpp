/*  0 <= n <= 4
         /\    /\    /\    /\    /\
        /11\  /13\  /15\  /17\  /19\     (10+2n+1) false
      /\----/\----/\----/\----/\----/
     / 0\10/ 2\12/ 4\14/ 6\16/ 8\18/     (2n or 10+2n) true / false
    /----\/----\/----\/----\/----\/
     \ 1/  \ 3/  \ 5/  \ 7/  \ 9/        (2n+1) true
      \/    \/    \/    \/    \/
*/

#ifndef __ICOSAHEDRON_FACES_H__
#define __ICOSAHEDRON_FACES_H__

#define NUM_OF_FACE 20

enum odd_even {
    EVEN = 0,
    ODD = 1
};

enum ele_area {
    BOTTOM = -2,
    MIDDLE_BOTTOM = -1,
    MIDDLE_TOP = 0,
    TOP = 1
};

class IcosaFaceTri;

class Icosahedron{
/*
Icosahedron is a graph of 20 IcosaFaceTri objects.

MEMBER VARIABLES:
    faces:              array of pointers to 20 IcosaFaceTri objects

METHODS:
    Icosahedron:        constructor
    ~Icosahedron:       destructor
    getFace4Angle:      returns face at cardinal directions(azimuth, elevation, angle)
*/
    private:
        IcosaFaceTri* faces[NUM_OF_FACE];

    public:
        Icosahedron();
        ~Icosahedron();
        void setFaceAttr();
        IcosaFaceTri* getStartFace(unsigned char azimuth, char elevation);
        IcosaFaceTri** getPovFaces(
            unsigned char *num_of_tris,
            IcosaFaceTri* startFace,
            unsigned char azimuth, char elevation, 
            unsigned char pov_x, unsigned char pov_y, unsigned char roll
        );
};

class IcosaFaceTri{
/* 
Faces of icosahedron are triangles.
This is graph model of one face.

MEMBER VARIABLES:
    faceId:             unique identifier of face
    triType:            false if face is type A ( Shape: /\ ), true if type B ( Shape: \/ )
    elevateFace:        pointer to face above or below this one
    leftFace:           pointer to face to the left of this one
    rightFace:          pointer to face to the right of this one

METHODS:
    IcosaFaceTri:       constructor
    ~IcosaFaceTri:      destructor
    getFaceId:          returns faceId
    getTriType:         returns triType
    getElevateFace:     returns elevateFace
    getLeftFace:        returns leftFace
    getRightFace:       returns rightFace
    getFaceFromAngle:   returns face at given angle
    setFaces:           sets elevateFace, leftFace, and rightFace
*/
    private:
        unsigned char       faceId;
        bool                triType;
        IcosaFaceTri        *elevateFace;
        IcosaFaceTri        *leftFace;
        IcosaFaceTri        *rightFace;

    public:
        IcosaFaceTri(unsigned char faceId, bool triType);
        ~IcosaFaceTri();
        void setCloserFaces(IcosaFaceTri* ele, IcosaFaceTri* left, IcosaFaceTri* right);
        unsigned char getFaceId();
        bool getTriType();
        IcosaFaceTri* getElevateFace();
        IcosaFaceTri* getLeftFace();
        IcosaFaceTri* getRightFace();
        IcosaFaceTri* getFaceFromAngle(unsigned short angle);
};

#endif /* __ICOSAHEDRON_FACES_H__ */