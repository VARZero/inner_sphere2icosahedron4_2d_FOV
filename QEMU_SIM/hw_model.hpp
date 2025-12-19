#ifndef __ICOSAHEDRON_FACES_H__
#define __ICOSAHEDRON_FACES_H__

enum odd_even {
    ODD = 0,
    EVEN = 1
} ODD_EVEN;

enum ele_area {
    BOTTOM = -2,
    MIDDLE_BOTTOM = -1,
    MIDDLE_TOP = 0,
    TOP = 1
} ELE_AREA;

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
        IcosaFaceTri* faces[20];

    public:
        Icosahedron();
        ~Icosahedron();
        IcosaFaceTri** getFaceAngle(
            unsigned char *num_of_tris,
            unsigned char azimuth, unsigned char elevation, 
            unsigned char pov_x, unsigned char pov_y, unsigned char roll
        );
}

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
        unsigned char getFaceId();
        bool getTriType();
        IcosaFaceTri* getElevateFace();
        IcosaFaceTri* getLeftFace();
        IcosaFaceTri* getRightFace();
        IcosaFaceTri* getFaceFromAngle(unsigned char angle);
        void setFaces(IcosaFaceTri *elevateFace, IcosaFaceTri *leftFace, IcosaFaceTri *rightFace);
}

#endif /* __ICOSAHEDRON_FACES_H__ */