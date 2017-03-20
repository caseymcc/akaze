#ifndef _akazeTypes_h_
#define _akazeTypes_h_

#include <vector>
#include "Eigen/Core"
namespace libAKAZE
{

/* ************************************************************************* */
// AKAZE Descriptor Type
enum DESCRIPTOR_TYPE
{
    SURF_UPRIGHT=0, // Upright descriptors, not invariant to rotation
    SURF=1,
    MSURF_UPRIGHT=2, // Upright descriptors, not invariant to rotation
    MSURF=3,
    MLDB_UPRIGHT=4, // Upright descriptors, not invariant to rotation
    MLDB=5
};

/* ************************************************************************* */
// AKAZE Diffusivities
enum DIFFUSIVITY_TYPE
{
    PM_G1=0,
    PM_G2=1,
    WEICKERT=2,
    CHARBONNIER=3
};

/* ************************************************************************* */
// Keypoint struct intended to mimic OpenCV.
struct Keypoint
{
//    float ptX;
//    float ptY;
    Eigen::Vector2f pt;
    float size;
    float angle;
    float response;
    int octave;
    int class_id;
#ifdef TRACK_REMOVED
    int removed;
#endif //TRACK_REMOVED
};

//// Descriptor type used for the float descriptors.
//struct Vector64f
//{
//    float operator[](size_t index) { return value[index]; }
//    float value[64];
//};
//typedef struct Vector64f Vector64f;
typedef Eigen::Matrix<float, 64, 1> Vector64f;
//
//// This convenience typdef is used to hold binary descriptors such that each bit
//// is a value in the descriptor.
//typedef std::vector<uint8_t> BinaryVectorX;

/* ************************************************************************* */
struct Descriptor
{
    void *m_data;
    size_t m_size;
};

struct Descriptors
{
    Descriptors() { m_binary=true; m_size=0; m_descriptorSize=0; }
    
    bool isBinary() const { return m_binary; }
    Descriptor &operator[](size_t index){return m_descriptors[index];}

    void binaryResize(size_t size, size_t descriptorSize)
    {
        m_descriptors.resize(size);
        binary_descriptor.resize(size*descriptorSize);

        for(size_t i=0; i<size; ++i)
        {
            m_descriptors[i].m_data=(void *)&binary_descriptor[i*descriptorSize];
            m_descriptors[i].m_size=descriptorSize;
        }

        m_binary=true;
        m_size=size;
        m_descriptorSize=descriptorSize;
    }
    void floatResize(size_t size, size_t descriptorSize)
    {
        float_descriptor.resize(size*descriptorSize);

        for(size_t i=0; i<size; ++i)
        {
            m_descriptors[i].m_data=(void *)&float_descriptor[i*descriptorSize];
            m_descriptors[i].m_size=descriptorSize;
        }

        m_binary=false;
        m_size=size;
        m_descriptorSize=descriptorSize;
    }

    void setZero()
    {
        if(m_binary)
        {
            for(auto &value:binary_descriptor)
                value=0;
        }
        else
        {
            for(auto &value:float_descriptor)
                value=0.0f;
        }

    }
    void *data() const {return m_binary?(void *)binary_descriptor.data():(void *)float_descriptor.data();}
    void *data(size_t index) const { return m_binary?(void *)&binary_descriptor[index*m_descriptorSize]:(void *)&float_descriptor[index*m_descriptorSize]; }
    const uint8_t *binaryData() const { return binary_descriptor.data(); }
    uint8_t *binaryData(){ return binary_descriptor.data(); }
    const uint8_t *binaryData(size_t index) const { return &binary_descriptor[index*m_descriptorSize]; }
    uint8_t *binaryData(size_t index) { return &binary_descriptor[index*m_descriptorSize]; }
    const float *floatData() const { return float_descriptor.data(); }
    const float *floatData(size_t index) const { return &float_descriptor[index*m_descriptorSize]; }
    float *floatData(size_t index){ return &float_descriptor[index*m_descriptorSize]; }
    size_t size() const { return m_size; }
    size_t descriptorSize() const { return m_descriptorSize; }
    
    std::vector<float> float_descriptor;
    std::vector<uint8_t> binary_descriptor;
    
    bool m_binary;
    size_t m_size;
    size_t m_descriptorSize;

    std::vector<Descriptor> m_descriptors;
};

struct Match
{
    float distance;
    int imgIdx;
    int queryIdx;
    int trainIdx;
};

}  // namespace libAKAZE

#endif  //_akazeTypes_h_
