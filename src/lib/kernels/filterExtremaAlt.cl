__kernel void findExtrema(read_only image2d_t input, int width, int height, __global float *extremaMap, int extremaMapWidth, int extremaMapHeight, int classId, int octave, float sigma, float threshold, float derivativeFactor, int offset)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int inputX=get_global_id(0);
    const int inputY=get_global_id(1);

    const int mapPixels=6;
    const int mapStride=extremaMapWidth*mapPixels;

//    float scaleX=(float)extremaMapWidth/width;
//    float scaleY=(float)extremaMapHeight/height;
//    int extremaX=floor(inputX*scaleX);
//    int extremaY=floor(intputY*scaleY);
    int scale=1;

    for(int i=1; i<octave; ++i)
        scale*=2;

    int extremaX=inputX*scale;
    int extremaY=inputY*scale;
    int pos=(extremaY*mapStride)+(extremaX*mapPixels);
    
    float value=read_imagef(input, nearestClampSampler, (int2)(inputX, inputY)).x;

    if((value<threshold) || (inputX < offset)||(inputY < offset) ||
        (inputX>= width-offset)||(inputY>= height-offset))
    {
        extremaMap[pos]=-1;
        extremaMap[pos+1]=-1;
        return;
    }

    int searchRadius=round(sigma*derivativeFactor/2);
    int yStart=inputY-searchRadius;
    int yEnd=inputY+searchRadius;
    int xStart=inputX-searchRadius;
    int xEnd=inputX+searchRadius;

    if(yStart<0)
        yStart=0;
    if(yEnd>=height)
        yStart=height-1;
    if(xStart<0)
        xStart=0;
    if(xEnd>=width)
        xStart=width-1;

    for(int y=yStart; y<=yEnd; ++y)
    {
        for(int x=xStart; x<=xEnd; ++x)
        {
            if((y==inputY) && (x==inputX))
                continue;

            float neighborValue=read_imagef(input, nearestClampSampler, (int2)(x, y)).x;

            if(neighborValue > value)
            {
                extremaMap[pos]=-1.0;
                extremaMap[pos+1]=-1.0;
                extremaMap[pos+2]=-1.0;
                return;
            }
        }
    }

    float2 delta=calculateSubPixel(input, inputX, inputY);

    if((delta.x<-1.0)||(delta.x>1.0)||(delta.y<-1.0)||(delta.y>1.0))
        return;

    extremaMap[pos]=(float)classId;
    extremaMap[pos+1]=value;
    extremaMap[pos+2]=searchRadius;
    extremaMap[pos+3]=delta.x;
    extremaMap[pos+4]=delta.y;
    
//    float value=read_imagef(input, nearestClampSampler, (int2)(outputX, outputY)).x;
//    float valueX_1Y_1=read_imagef(input, nearestClampSampler, (int2)(outputX-1, outputY-1)).x;
//    float valueX_1Y=read_imagef(input, nearestClampSampler, (int2)(outputX-1, outputY)).x;
//    float valueX_1Y1=read_imagef(input, nearestClampSampler, (int2)(outputX-1, outputY+1)).x;
//    float valueXY_1=read_imagef(input, nearestClampSampler, (int2)(outputX, outputY-1)).x;
//    float valueXY1=read_imagef(input, nearestClampSampler, (int2)(outputX, outputY+1)).x;
//    float valueX1Y_1=read_imagef(input, nearestClampSampler, (int2)(outputX+1, outputY-1)).x;
//    float valueX1Y=read_imagef(input, nearestClampSampler, (int2)(outputX+1, outputY)).x;
//    float valueX1Y1=read_imagef(input, nearestClampSampler, (int2)(outputX+1, outputY+1)).x;
//    float mapClass=-1.0;
//    
//    // Filter the points with the detector threshold
//    if(value>threshold &&
//        value>valueXY_1&&value>valueXY1&&
//        value>valueX_1Y_1&&value>valueX_1Y&&
//        value>valueX_1Y1&&value>valueX1Y_1&&
//        value>valueX1Y&&value>valueX1Y1)
//    {
//        mapClass=(float)classId;
//    }
//    else
//        value=-1.0;
//
//    extremaMap[pos]=mapClass;
//    extremaMap[pos+1]=value;
}

__kernel void findExtremaIterate(read_only image2d_t input, int width, int height, __global float *extremaMap, int extremaMapWidth, int extremaMapHeight, int classId, int octave, float sigma, float threshold, float derivativeFactor, int offset)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int inputX=get_global_id(0)+offset;
    const int inputY=get_global_id(1)+offset;

    float value=read_imagef(input, nearestClampSampler, (int2)(inputX, inputY)).x;

    if(value<threshold)
        return;

    const int mapPixels=6;
    const int mapStride=extremaMapWidth*mapPixels;

//    float scaleX=(float)extremaMapWidth/width;
//    float scaleY=(float)extremaMapHeight/height;
//    int extremaX=floor(outputX*scaleX);
//    int extremaY=floor(outputY*scaleY);
    int scale=1;

    for(int i=1; i<octave; ++i)
        scale*=2;

    int extremaX=inputX*scale;
    int extremaY=inputY*scale;

//    float value=read_imagef(input, nearestClampSampler, (int2)(outputX, outputY)).x;
//    float valueX_1Y_1=read_imagef(input, nearestClampSampler, (int2)(outputX-1, outputY-1)).x;
//    float valueX_1Y=read_imagef(input, nearestClampSampler, (int2)(outputX-1, outputY)).x;
//    float valueX_1Y1=read_imagef(input, nearestClampSampler, (int2)(outputX-1, outputY+1)).x;
//    float valueXY_1=read_imagef(input, nearestClampSampler, (int2)(outputX, outputY-1)).x;
//    float valueXY1=read_imagef(input, nearestClampSampler, (int2)(outputX, outputY+1)).x;
//    float valueX1Y_1=read_imagef(input, nearestClampSampler, (int2)(outputX+1, outputY-1)).x;
//    float valueX1Y=read_imagef(input, nearestClampSampler, (int2)(outputX+1, outputY)).x;
//    float valueX1Y1=read_imagef(input, nearestClampSampler, (int2)(outputX+1, outputY+1)).x;
    int searchRadius=round(sigma*derivativeFactor/2);
    int yStart=inputY-searchRadius;
    int yEnd=inputY+searchRadius;
    int xStart=inputX-searchRadius;
    int xEnd=inputX+searchRadius;

    if(yStart<0)
        yStart=0;
    if(yEnd>=height)
        yStart=height-1;
    if(xStart<0)
        xStart=0;
    if(xEnd>=width)
        xStart=width-1;

    for(int y=yStart; y<=yEnd; ++y)
    {
        for(int x=xStart; x<=xEnd; ++x)
        {
            if((y==inputY)&&(x==inputX))
                continue;

            float neighborValue=read_imagef(input, nearestClampSampler, (int2)(x, y)).x;

            if(neighborValue > value)
                return;
        }
    }

//    // Filter the points with the detector threshold
////    if(value>threshold &&
////        value>valueXY_1&&value>valueXY1&&
////        value>valueX_1Y_1&&value>valueX_1Y&&
////        value>valueX_1Y1&&value>valueX1Y_1&&
////        value>valueX1Y&&value>valueX1Y1)
//    {
////        int searchRadius=floor(sigma*derivativeFactor/2);
//        int yExtremaStart=extremaY-searchRadius;
//        int yExtremaEnd=extremaY+searchRadius;
//        int xExtremaStart=extremaX-searchRadius;
//        int xExtremaEnd=extremaX+searchRadius;
//
//        if(yExtremaStart<0)
//            yExtremaStart=0;
//        if(yExtremaEnd>=extremaMapHeight)
//            yExtremaStart=extremaMapHeight-1;
//        if(xExtremaStart<0)
//            xExtremaStart=0;
//        if(xExtremaEnd>=extremaMapWidth)
//            xExtremaStart=extremaMapWidth-1;
//
//        //search for other local extrema
//        for(int y=yExtremaStart; y<=yExtremaEnd; ++y)
//        {
//            int mapPosY=mapStride*y;
//
//            for(int x=xExtremaStart; x<=xExtremaEnd; ++x)
//            {
//                int pos=mapPosY+(x*mapPixels);
//
//                if(extremaMap[pos] >= 0.0)
//                {
//                    //another extrema point found with higher response
//                    if(extremaMap[pos+1]>value)
//                        return;
//                }
//            }
//        }
//
//        //add extrema
//        int pos=(extremaY*mapStride)+(extremaX*mapPixels);
//
//        extremaMap[pos]=classId;
//        extremaMap[pos+1]=value;
//    }

    int pos=(extremaY*mapStride)+(extremaX*mapPixels);

    //check if lower extrema is higher value
    if(extremaMap[pos] >= 0)
    {
        if(extremaMap[pos+1]>value)
            return;
    }

    
    float2 delta=calculateSubPixel(input, inputX, inputY);

    if((delta.x<-1.0)||(delta.x>1.0)||(delta.y<-1.0)||(delta.y>1.0))
        return;

    extremaMap[pos]=(float)classId;
    extremaMap[pos+1]=value;
    extremaMap[pos+2]=searchRadius;
    extremaMap[pos+3]=delta.x;
    extremaMap[pos+4]=delta.y;
}

float2 calculateSubPixelBuffer(__global float *input, int offset, int width, int height, int x, int y)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;

    float2 delta;
    int pos=offset+(y*width)+x;

    float value=input[pos];
    float valueX_1Y_1=input[pos-width-1];
    float valueX_1Y=input[pos-1];
    float valueX_1Y1=input[pos+width-1];
    float valueXY_1=input[pos-width];
    float valueXY1=input[pos+width];
    float valueX1Y_1=input[pos-width+1];
    float valueX1Y=input[pos+1];
    float valueX1Y1=input[pos+width+1];

    // Compute the gradient
    float dx=0.5*(valueX1Y-valueX_1Y);
    float dy=0.5*(valueXY1-valueXY_1);

    // Compute the Hessian
    float dxx=valueX1Y+valueX_1Y-2.0*value;
    float dyy=valueXY1+valueXY_1-2.0*value;
    float dxy=(0.25*(valueX1Y1+valueX_1Y_1))-(0.25*(valueX1Y_1+valueX_1Y1));

    //solve Ax=B
    float det=(dxx*dyy)-(dxy*dxy);
    float invDet=0.0f;

    if(det!=0.0)
        invDet=1.0f/det;

    delta.x=invDet*(dxy*dy-dyy*dx);
    delta.y=invDet*(dxy*dx-dxx*dy);

    return delta;
}

struct __attribute__((aligned(16))) Keypoint
{
    float ptX;
    float ptY;
    float size;
    float angle;
    float response;
    int octave;
    int class_id;
};
typedef struct Keypoint Keypoint;

__kernel void findExtremaBuffer(__global float *input, int inputOffset, int width, int height, __global float *extremaMap, int extremaMapWidth, int extremaMapHeight, int classId, int octave, float sigma, float threshold, float derivativeFactor, int offset)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int inputX=get_global_id(0);
    const int inputY=get_global_id(1);

    const int mapPixels=6;
    const int mapStride=extremaMapWidth*mapPixels;
    int scale=1;
    int inputPos=inputOffset+(inputY*width)+inputX;

    for(int i=1; i<octave; ++i)
        scale*=2;

    int extremaX=inputX*scale;
    int extremaY=inputY*scale;
    int pos=(extremaY*mapStride)+(extremaX*mapPixels);

    float value=input[inputPos];

    if((value<threshold)||(inputX < offset)||(inputY < offset)||
        (inputX>=width-offset)||(inputY>=height-offset))
    {
        extremaMap[pos]=-1;
        extremaMap[pos+1]=-1;
        return;
    }

    int searchRadius=round(sigma*derivativeFactor/2);
    int yStart=inputY-searchRadius;
    int yEnd=inputY+searchRadius;
    int xStart=inputX-searchRadius;
    int xEnd=inputX+searchRadius;

    if(yStart<0)
        yStart=0;
    if(yEnd>=height)
        yStart=height-1;
    if(xStart<0)
        xStart=0;
    if(xEnd>=width)
        xStart=width-1;

    for(int y=yStart; y<=yEnd; ++y)
    {
        inputPos=inputOffset+(y*width);
        for(int x=xStart; x<=xEnd; ++x)
        {
            if((y==inputY)&&(x==inputX))
                continue;

            float neighborValue=input[inputPos+x];

            if(neighborValue > value)
            {
                extremaMap[pos]=-1.0;
                extremaMap[pos+1]=-1.0;
//                extremaMap[pos+2]=-1.0;
                return;
            }
        }
    }

    float2 delta=calculateSubPixelBuffer(input, inputOffset, width, height, inputX, inputY);

    if((delta.x<-1.0)||(delta.x>1.0)||(delta.y<-1.0)||(delta.y>1.0))
        return;

    extremaMap[pos]=(float)classId;
    extremaMap[pos+1]=value;
    extremaMap[pos+2]=searchRadius;
    extremaMap[pos+3]=delta.x;
    extremaMap[pos+4]=delta.y;
    extremaMap[pos+5]=octave;
}

__kernel void findExtremaIterateBuffer(__global float *input, int inputOffset, int width, int height, __global float *extremaMap, int extremaMapWidth, int extremaMapHeight, int classId, int octave, float sigma, float threshold, float derivativeFactor, int offset)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int inputX=get_global_id(0)+offset;
    const int inputY=get_global_id(1)+offset;
    int inputPos=inputOffset+(inputY*width)+inputX;

    float value=input[inputPos];

    if(value<threshold)
        return;

    const int mapPixels=6;
    const int mapStride=extremaMapWidth*mapPixels;
    int scale=1;

    for(int i=1; i<octave; ++i)
        scale*=2;

    int extremaX=inputX*scale;
    int extremaY=inputY*scale;

    float pointSize=sigma*derivativeFactor;
    float squaredPointSize=pointSize*pointSize;
    int searchRadius=ceil(pointSize);
    int yStart=inputY-searchRadius;
    int yEnd=inputY+searchRadius;
    int xStart=inputX-searchRadius;
    int xEnd=inputX+searchRadius;

    if(yStart<0)
        yStart=0;
    if(yEnd>=height)
        yStart=height-1;
    if(xStart<0)
        xStart=0;
    if(xEnd>=width)
        xStart=width-1;

    for(int y=yStart; y<=yEnd; ++y)
    {
        inputPos=inputOffset+(y*width);
        for(int x=xStart; x<=xEnd; ++x)
        {
            if((y==inputY)&&(x==inputX))
                continue;

            float deltaX=(x*scale)-inputX;
            float deltaY=(y*scale)-inputY;

            float squaredDist=(deltaX*deltaX)+(deltaY*deltaY);

            if(squaredDist>squaredPointSize)
                continue;

            float neighborValue=input[inputPos+x];

            if(neighborValue > value)
                return;
        }
    }

    int pos=(extremaY*mapStride)+(extremaX*mapPixels);

    //check if lower extrema is higher value
    if(extremaMap[pos]>=0)
    {
        if(extremaMap[pos+1]>value)
            return;
    }


    float2 delta=calculateSubPixelBuffer(input, inputOffset, width, height, inputX, inputY);

    if((delta.x<-1.0)||(delta.x>1.0)||(delta.y<-1.0)||(delta.y>1.0))
        return;

    extremaMap[pos]=(float)classId;
    extremaMap[pos+1]=value;
    extremaMap[pos+2]=pointSize;
    extremaMap[pos+3]=delta.x;
    extremaMap[pos+4]=delta.y; 
    extremaMap[pos+5]=octave;
}

__kernel void filterExtrema(__global float *extremaMap, int width, int height, int offset, __global int *count)
{
    // currently will not filter points that are within a distant to a higher value point that includes the lower value point in its search range,
    // but that the lower value point search does not included the higher value point. This should be fixable by flagging the lower value from the higher
    // values search but this needs to implement local memory atomics to "properly handle it"
    __local int workGroupCount;

    if((get_local_id(0))==0&&(get_local_id(1)==0))
        workGroupCount=0;

    barrier(CLK_LOCAL_MEM_FENCE);

    const int inputX=get_global_id(0)+offset;
    const int inputY=get_global_id(1)+offset;

    const int mapPixels=6;
    const int mapStride=width*mapPixels;
    int pos=(inputY*mapStride)+(inputX*mapPixels);
    int classId=extremaMap[pos];
    
    if(classId>=0)
    {
        int valid=1;
        int searchRadius=extremaMap[pos+2];
        int yStart=inputY-searchRadius;
        int yEnd=inputY+searchRadius;
        int xStart=inputX-searchRadius;
        int xEnd=inputX+searchRadius;

        if(yStart<0)
            yStart=0;
        if(yEnd>=height)
            yStart=height-1;
        if(xStart<0)
            xStart=0;
        if(xEnd>=width)
            xStart=width-1;

        float value=extremaMap[pos+1];
        for(int y=yStart; y<=yEnd; ++y)
        {
            int neighborPosY=y*mapStride;
            for(int x=xStart; x<=xEnd; ++x)
            {
                if((y==inputY)&&(x==inputX))
                    continue;

                if(extremaMap[neighborPosY+(x*mapPixels)]<0)
                    continue;

                float neighborValue=extremaMap[neighborPosY+(x*mapPixels)+1];

                if(neighborValue>value)
                {
                    extremaMap[pos]=-1;
                    extremaMap[pos+1]=-1;
                    extremaMap[pos+2]=-1;
//                    return;
                    y=yEnd+1;
                    valid=0;
                    break;
                }
            }
        }

        if(valid>0)
            atomic_inc(&workGroupCount);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if((get_local_id(0))==0&&(get_local_id(1)==0))
        atomic_add(count, workGroupCount);

//    int index=atomic_inc(count);
//
//    if(index>=maxKeypoints)
//        return;
//
//    Keypoint &keypoint=keypoints[index];
//
//    keypoint.ptX=inputX+extremaMap[pos+3];
//    keypoint.ptY=inputY+extremaMap[pos+4];
//    keypoint.size=searchRadius;
//    keypoint.angle=0.0;
//    keypoint.response=value;
//    keypoint.octave=extremaMap[pos+5];
//    keypoint.class_id=classId;
}

__kernel void consolidateKeypoints(__global float *extremaMap, int width, int height, int offset, __global Keypoint *keypoints, int maxKeypoints, __global int *count)
{
    const int inputX=get_global_id(0)+offset;
    const int inputY=get_global_id(1)+offset;

    const int mapPixels=6;
    const int mapStride=width*mapPixels;
    int pos=(inputY*mapStride)+(inputX*mapPixels);
    int classId=extremaMap[pos];

    if(classId<0)
        return;

    int index=atomic_inc(count);

    if(index>=maxKeypoints)
        return;

    __global Keypoint *keypoint=&keypoints[index];

    keypoint->ptX=inputX+extremaMap[pos+3];
    keypoint->ptY=inputY+extremaMap[pos+4];
    keypoint->size=extremaMap[pos+2];
    keypoint->angle=0.0;
    keypoint->response=extremaMap[pos+1];
    keypoint->octave=extremaMap[pos+5];
    keypoint->class_id=classId;
}