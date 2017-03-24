//#define TRACK_REMOVED

__kernel void convolve(read_only image2d_t input, __constant float *kernelBuffer, const int kernelSize, write_only image2d_t output)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int xOutput=get_global_id(0);
    const int yOutput=get_global_id(1);
    float sum=0;

    int filterOffset=kernelSize/2;
    int xInput=xOutput-filterOffset;
    int yInput=yOutput-filterOffset;

    for(int row=0; row<kernelSize; row++)
    {
        const int indexFilterRow=row*kernelSize;
        const int yInputRow=yInput+row;

        for(int col=0; col<kernelSize; col++)
        {
            const int indexFilter=indexFilterRow+col;
            float value=read_imagef(input, nearestClampSampler, (int2)(xInput+col, yInputRow)).x;

            sum+=kernelBuffer[indexFilter]*value;
        }
    }
    write_imagef(output, (int2)(xOutput, yOutput), sum);
}

__kernel void separableConvolveXImage2D(read_only image2d_t input, __constant float *kernelX, const int kernelSize, float scale, write_only image2d_t output)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;

    const int xOutput=get_global_id(0);
    const int yOutput=get_global_id(1);

    float sum=0;
    int filterOffset=kernelSize/2;
    int xInput=xOutput-filterOffset;

    for(int x=0; x<kernelSize; x++)
    {
        float value=read_imagef(input, nearestClampSampler, (int2)(xInput+x, yOutput)).x;

        sum+=kernelX[x]*value;
    }

    sum=sum*scale;
    write_imagef(output, (int2)(xOutput, yOutput), sum);
}

//__kernel void separableConvolveXImage2D(read_only image2d_t input, int width, __constant float *kernelX, const int kernelSize, float scale, write_only image2d_t output)
//{
//    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
//
//    const int xOutput=get_global_id(0);
//
//    float sum=0;
//    int filterOffset=kernelSize/2;
//    int xInput=xOutput-filterOffset;
//
//    __local float cache[32];
//    int pos=0;
//
//    for(int i=0; i<filterOffset; i++)
//        cache[pos]=read_imagef(input, nearestClampSampler, (int2)(xInput+x, yOutput)).x;
//
//    for(int x=0; x<kernelSize; x++)
//    {
//        float value=read_imagef(input, nearestClampSampler, (int2)(xInput+x, yOutput)).x;
//
//        sum+=kernelX[x]*value;
//    }
//
//    sum=sum*scale;
//    write_imagef(output, (int2)(xOutput, yOutput), sum);
//}

__kernel void separableConvolveXImage2DBuffer(read_only image2d_t input, __constant float *kernelX, const int kernelSize, float scale, __global float *output, int offset, int width, int height)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;

    const int xOutput=get_global_id(0);
    const int yOutput=get_global_id(1);

    float sum=0;
    int filterOffset=kernelSize/2;
    int xInput=xOutput-filterOffset;
    int outputPos=offset+(yOutput*width)+xOutput;

    for(int x=0; x<kernelSize; x++)
    {
        float value=read_imagef(input, nearestClampSampler, (int2)(xInput+x, yOutput)).x;

        sum+=kernelX[x]*value;
    }

    sum=sum*scale;
    output[outputPos]=sum;
}

__kernel void separableConvolveXBuffer(__global float *input, int inputOffset, int width, int height, __constant float *kernelBuffer, const int kernelSize, float scale, __global float *output, int outputOffset)
{
    const int xOutput=get_global_id(0);
    const int yOutput=get_global_id(1);

    float sum=0;
    int kSize=kernelSize;
    const int filterOffset=kernelSize/2;
    const int intputPos=inputOffset+(yOutput*width)+(xOutput-filterOffset);
    const int outputPos=outputOffset+(yOutput*width)+xOutput;

    int imageStart=xOutput-filterOffset;
    int imageEnd=xOutput+filterOffset;
    int kernelStart=0;

    if(imageStart < 0)
    {
        int shift=-imageStart;
        int pos=intputPos+shift;

        for(int i=0; i<shift; i++)
            sum+=kernelBuffer[i]*input[pos];

        kernelStart=shift;
    }

    if(imageEnd >= width)
    {
        int shift=imageEnd-width+1;
        int pos=intputPos+(kernelSize-shift-1);

        for(int i=kernelSize-shift; i<kernelSize; i++)
            sum+=kernelBuffer[i]*input[pos];

        kSize-=shift;
    }

    for(int i=kernelStart; i<kSize; i++)
        sum+=kernelBuffer[i]*input[intputPos+i];

    sum=sum*scale;
    output[outputPos]=sum;
}

__kernel void separableConvolveYImage2D(read_only image2d_t input, __constant float *kernelY, const int kernelSize, float scale, write_only image2d_t output)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;

    const int xOutput=get_global_id(0);
    const int yOutput=get_global_id(1);

    float sum=0;
    int kSize=kernelSize;
    int filterOffset=kernelSize/2;
    int yInput=yOutput-filterOffset;

    for(int y=0; y<kernelSize; y++)
    {
        float value=read_imagef(input, nearestClampSampler, (int2)(xOutput, yInput+y)).x;
        
        sum+=kernelY[y]*value;
    }

    sum=sum*scale;
    write_imagef(output, (int2)(xOutput, yOutput), sum);
}

__kernel void separableConvolveYBuffer(__global float *input, int inputOffset, int width, int height, __constant float *kernelBuffer, const int kernelSize, float scale, __global float *output, int outputOffset)
{
    const int xOutput=get_global_id(0);
    const int yOutput=get_global_id(1);

    float sum=0;
    const int filterOffset=kernelSize/2;
    const int intputPos=inputOffset+((yOutput-filterOffset)*width)+xOutput;
    const int outputPos=outputOffset+(yOutput*width)+xOutput;

    int imageStart=yOutput-filterOffset;
    int imageEnd=yOutput+filterOffset;
    int kernelStart=0;
    int kSize=kernelSize;

    if(imageStart<0)
    {
        int shift=-imageStart;
        int pos=intputPos+(shift*width);

        for(int i=0; i<shift; i++)
            sum+=kernelBuffer[i]*input[pos];

        kernelStart=shift;
    }

    if(imageEnd>=height)
    {
        int shift=imageEnd-height+1;
        int pos=intputPos+((kernelSize-shift-1)*width);

        for(int i=kernelSize-shift; i<kernelSize; i++)
            sum+=kernelBuffer[i]*input[pos];

        kSize-=shift;
    }

    for(int i=kernelStart; i<kSize; i++)
        sum+=kernelBuffer[i]*input[intputPos+(i*width)];

    sum=sum*scale;
    output[outputPos]=sum;
}

__kernel void magnitude(read_only image2d_t input1, read_only image2d_t input2, write_only image2d_t output)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int xOutput=get_global_id(0);
    const int yOutput=get_global_id(1);

    float value1=read_imagef(input1, nearestClampSampler, (int2)(xOutput, yOutput)).x;
    float value2=read_imagef(input2, nearestClampSampler, (int2)(xOutput, yOutput)).x;

    value1=sqrt(value1*value1+value2*value2);

    write_imagef(output, (int2)(xOutput, yOutput), value1);
}

__kernel void determinantHessian(read_only image2d_t dxx, read_only image2d_t dyy, read_only image2d_t dxy, write_only image2d_t output)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int xOutput=get_global_id(0);
    const int yOutput=get_global_id(1);

    float value1=read_imagef(dxx, nearestClampSampler, (int2)(xOutput, yOutput)).x;
    float value2=read_imagef(dyy, nearestClampSampler, (int2)(xOutput, yOutput)).x;
    float value3=read_imagef(dxy, nearestClampSampler, (int2)(xOutput, yOutput)).x;

//    float value=sqrt(value1*value2)-(value3*value3);
    float value=(value1*value2)-(value3*value3);

    write_imagef(output, (int2)(xOutput, yOutput), value);
}

__kernel void determinantHessianBuffer(__global float *dxx, int dxxOffset, __global float *dyy, int dyyOffset, __global float *dxy, int dxyOffset, int width, int height, __global float *output, int outputOffset)
{
    const int xOutput=get_global_id(0);
    const int yOutput=get_global_id(1);

    int pos=(yOutput*width)+xOutput;

    float value1=dxx[pos+dxxOffset];
    float value2=dyy[pos+dyyOffset];
    float value3=dxy[pos+dxyOffset];

    output[pos+outputOffset]=(value1*value2)-(value3*value3);
}

__kernel void rowMax(read_only image2d_t input, int width, __global float *output)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int index=get_global_id(0);

    float max=read_imagef(input, nearestClampSampler, (int2)(0, index)).x;
    for(int i=1; i<width; i++)
    {
        float value=read_imagef(input, nearestClampSampler, (int2)(i, index)).x;

        if(value>max)
            max=value;
    }

    output[index]=max;
}

__kernel void histogramRows(read_only image2d_t input, int width, int height, int bins, float scale, __global int *output)//, volatile __local int *accumulator)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int index=get_global_id(0);

    if(index>=height)
        return;

    const int stride=bins*index;
//    const int accumulatorStride=bins*get_local_id(0);

    for(int i=0; i<bins; i++)
    {
//        accumulator[accumulatorStride+i]=0;
        output[stride+i]=0;
    }

    for(int i=0; i<width; i++)
    {
        float value=read_imagef(input, nearestClampSampler, (int2)(i, index)).x;
        int bin=floor(bins*(value*scale));
//
        if(bin>=bins)
            bin=bins-1;
        else if(bin<0)
            bin=0;

//        int count=output[stride+bin];
//
//        count++;
//        output[stride+bin]=count;
        output[stride+bin]++;
////        int count=accumulator[accumulatorStride+bin];
////
////        count++;
////        accumulator[accumulatorStride+bin]=count;
//        accumulator[accumulatorStride+bin]=value;
    }

//    for(int i=0; i<bins; i++)
//        output[stride+i]=accumulator[accumulatorStride+i];
}

__kernel void histogramCombine(__global int *input, int bins, int count, __global int *output)
{
    const int bin=get_global_id(0);
//    const int stride=bins*bin;
    int value=0;

    for(int i=0; i<count; i++)
        value+=input[bins*i+bin];

    output[bin]=value;
}

__kernel void linearSample(read_only image2d_t input, write_only image2d_t output, int outputWidth, int outputHeight)
{
    const sampler_t linearSampler=CLK_NORMALIZED_COORDS_TRUE|CLK_FILTER_LINEAR|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int outputX=get_global_id(0);
    const int outputY=get_global_id(1);

    float inputX=((float)outputX+0.5)/outputWidth;
    float inputY=((float)outputY+0.5)/outputHeight;

    float value=read_imagef(input, linearSampler, (float2)(inputX, inputY)).x;
    
    write_imagef(output, (int2)(outputX, outputY), value);
}

__kernel void pmG1(read_only image2d_t input1, read_only image2d_t input2, write_only image2d_t output, float k)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int outputX=get_global_id(0);
    const int outputY=get_global_id(1);
    const float inv_k=1.0/(k * k);

    float value1=read_imagef(input1, nearestClampSampler, (int2)(outputX, outputY)).x;
    float value2=read_imagef(input2, nearestClampSampler, (int2)(outputX, outputY)).x;

    float value=exp(-inv_k*(value1*value1+value2*value2));
    write_imagef(output, (int2)(outputX, outputY), value);
}

__kernel void pmG2(read_only image2d_t input1, read_only image2d_t input2, write_only image2d_t output, float k)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int outputX=get_global_id(0);
    const int outputY=get_global_id(1);
    const float inv_k=1.0/(k * k);

    float value1=read_imagef(input1, nearestClampSampler, (int2)(outputX, outputY)).x;
    float value2=read_imagef(input2, nearestClampSampler, (int2)(outputX, outputY)).x;

    float value=1.0/(1.0+inv_k*(value1*value1+value2*value2));
    write_imagef(output, (int2)(outputX, outputY), value);
}

__kernel void weickert(read_only image2d_t input1, read_only image2d_t input2, write_only image2d_t output, float k)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int outputX=get_global_id(0);
    const int outputY=get_global_id(1);
    const float inv_k=1.0/(k * k);

    float value1=read_imagef(input1, nearestClampSampler, (int2)(outputX, outputY)).x;
    float value2=read_imagef(input2, nearestClampSampler, (int2)(outputX, outputY)).x;

    float diff=inv_k*(value1*value1+value2*value2);
    float value=1.0-exp(-3.315/(diff*diff*diff*diff));
    write_imagef(output, (int2)(outputX, outputY), value);
}

__kernel void charbonnier(read_only image2d_t input1, read_only image2d_t input2, write_only image2d_t output, float k)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int outputX=get_global_id(0);
    const int outputY=get_global_id(1);
    const float inv_k=1.0/(k * k);

    float value1=read_imagef(input1, nearestClampSampler, (int2)(outputX, outputY)).x;
    float value2=read_imagef(input2, nearestClampSampler, (int2)(outputX, outputY)).x;

    float value=1.0/sqrt(1.0+inv_k*(value1*value1+value2*value2));
    write_imagef(output, (int2)(outputX, outputY), value);
}

__kernel void nldStepScalar(read_only image2d_t input, read_only image2d_t flow, write_only image2d_t output, float stepsize)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;
    const int outputX=get_global_id(0);
    const int outputY=get_global_id(1);

    float inputValue=read_imagef(input, nearestClampSampler, (int2)(outputX, outputY)).x;
    float inputValueX1=read_imagef(input, nearestClampSampler, (int2)(outputX+1, outputY)).x;
    float inputValueX_1=read_imagef(input, nearestClampSampler, (int2)(outputX-1, outputY)).x;
    float inputValueY1=read_imagef(input, nearestClampSampler, (int2)(outputX, outputY+1)).x;
    float inputValueY_1=read_imagef(input, nearestClampSampler, (int2)(outputX, outputY-1)).x;

    float flowValue=read_imagef(flow, nearestClampSampler, (int2)(outputX, outputY)).x;
    float flowValueX1=read_imagef(flow, nearestClampSampler, (int2)(outputX+1, outputY)).x;
    float flowValueX_1=read_imagef(flow, nearestClampSampler, (int2)(outputX-1, outputY)).x;
    float flowValueY1=read_imagef(flow, nearestClampSampler, (int2)(outputX, outputY+1)).x;
    float flowValueY_1=read_imagef(flow, nearestClampSampler, (int2)(outputX, outputY-1)).x;

    const float xpos=(flowValue+flowValueX1) * (inputValueX1-inputValue);
    const float xneg=(flowValueX_1+flowValue) * (inputValue-inputValueX_1);
    const float ypos=(flowValue+flowValueY1) * (inputValueY1-inputValue);
    const float yneg=(flowValueY_1+flowValue) * (inputValue-inputValueY_1);

    float value=0.5*stepsize*(xpos-xneg+ypos-yneg)+inputValue;

    write_imagef(output, (int2)(outputX, outputY), value);
}


__kernel void scale(__global float *input, float scaleValue, int items)
{ 
    const int index=get_global_id(0);

    if(index>=items)
        return;

    input[index]=input[index]*scaleValue;
}

struct __attribute__((aligned(16))) EvolutionInfo
{
    int width;
    int height;
    int offset;
    float sigma;
    float pointSize;
    int octave;
};
typedef struct EvolutionInfo EvolutionInfo;

struct __attribute__((aligned(16))) Keypoint
{
    int class_id;
    int octave;
#ifdef TRACK_REMOVED
    int removed;
#endif //TRACK_REMOVED
    float size;
    float ptX;
    float ptY;
    float response;
    float angle;
};
typedef struct Keypoint Keypoint;

struct __attribute__((aligned(8))) ExtremaMap
{
    int class_id;
    float response;
};
typedef struct ExtremaMap ExtremaMap;

__kernel void findExtremaBuffer(__global float *input, volatile __global ExtremaMap *extremaMap, int mapIndex, __global EvolutionInfo *evolutionBuffer, float threshold, int workOffset, __global int *keypointCount, int maxX, int maxY)
{
    volatile __local int workGroupCount;
    const int localX=get_local_id(0);
    const int localY=get_local_id(1);

    const int globalX=get_global_id(0);
    const int globalY=get_global_id(1);

    if((mapIndex == 0) &&(globalX == 0) && (globalY == 0))
        *keypointCount=0;
    if((localX == 0) && (localY == 0))
        workGroupCount=0;

    barrier(CLK_LOCAL_MEM_FENCE);

//    //ignore items heigher than working area
//    if((globalX >= maxX) || (globalY >= maxY))
//        return;

    const int inputX=globalX+workOffset;
    const int inputY=globalY+workOffset;

    __global EvolutionInfo *evolution=&evolutionBuffer[mapIndex];

    const float sigma=evolution->sigma;
    const int octave=evolution->octave;
    const int offset=evolution->offset;
    const int width=evolution->width;
    const int height=evolution->height;
    const float pointSize=evolution->pointSize;

    int inputPos=offset+(inputY*width)+inputX;
    float scale=(float)(1<<octave);

    int pos=offset+(inputY*width)+inputX;
    float value=input[inputPos];

    if(value<threshold)
    {
        extremaMap[pos].class_id=-1;
    }
	else
	{
		float size=pointSize;
		int searchRadius=ceil(size/scale); //alter search by scale
		int yStart=inputY-searchRadius;
		int yEnd=inputY+searchRadius;
		int xStart=inputX-searchRadius;
		int xEnd=inputX+searchRadius;

		float squaredSize=size*size; //square for comparisons

		if(yStart<0)
			yStart=0;
		if(yEnd>=height)
			yStart=height-1;
		if(xStart<0)
			xStart=0;
		if(xEnd>=width)
			xStart=width-1;

		float pointX=(inputX)*scale;
		float pointY=(inputY)*scale;

		bool extrema=true;

		//search current frame for extrema
		for(int y=yStart; y<=yEnd; ++y)
		{
			inputPos=offset+(y*width);
			for(int x=xStart; x<=xEnd; ++x)
			{
				if((y==inputY)&&(x==inputX))
					continue;

				float nPointX=pointX-(x*scale);
				float nPointY=pointY-(y*scale);
				float dist=(nPointX*nPointX)+(nPointY*nPointY);

				if(dist>squaredSize)
					continue;

				float neighborValue=input[inputPos+x];

				if(neighborValue>value)
				{
					extremaMap[pos].class_id=-1;
					y = yEnd + 1;
					extrema=false;
					break;
				}
			}
		}

		if(mapIndex > 0)
		{
			//search previous frame for duplicate

			__global EvolutionInfo *prevEvolution=&evolutionBuffer[mapIndex-1];

			const float prevSigma=prevEvolution->sigma;
			const int prevOctave=prevEvolution->octave;
			const int prevOffset=prevEvolution->offset;
            const int prevWidth=prevEvolution->width;
            const int prevHeight=prevEvolution->height;

			float prevScale=(float)(1<<prevOctave);

			int prevSearchRadius=ceil(size/prevScale); //alter search by scale
			int yStart=(inputY*scale)-searchRadius;
			int yEnd=(inputY*scale)+searchRadius;
			int xStart=(inputX*scale)-searchRadius;
			int xEnd=(inputX*scale)+searchRadius;

			for(int y=yStart; y<=yEnd; ++y)
			{
                int prevPos=prevOffset+(y*prevWidth);

				for(int x=xStart; x<=xEnd; ++x)
				{
					float prevPointX=pointX-(x*prevScale);
					float prevPointY=pointY-(y*prevScale);
					float dist=(prevPointX*prevPointX)+(prevPointY*prevPointY);

					if(dist > squaredSize)
						continue;

                    if(extremaMap[prevPos+x].class_id < 0)
                        continue;

					float prevValue=input[prevPos+x];

					if(prevValue > value)
					{
#ifdef TRACK_REMOVED
                        mapIndex=mapIndex+16;
#else //TRACK_REMOVED
						extremaMap[pos].class_id=-1;
						extrema=false;
#endif //TRACK_REMOVED
                        y=yEnd+1;
						break;
					}
					else //we have higher value knock this one out
					{
#ifdef TRACK_REMOVED
                        int prevClassId=extremaMap[prevPos+x].class_id;
                        
                        if(prevClassId < 16)
                            atomic_cmpxchg(&extremaMap[prevPos+x].class_id, prevClassId, prevClassId+32);
#else //TRACK_REMOVED
                        int oldValue=atomic_xchg(&extremaMap[prevPos+x].class_id, -1);

                        if(oldValue != -1)
						    atomic_dec(&workGroupCount); //just removed a previous point
#endif //TRACK_REMOVED
					}
				}
			}
		}

		if(extrema)
		{
			extremaMap[pos].class_id=mapIndex;
			extremaMap[pos].response=value;
			atomic_inc(&workGroupCount);
		}
	}

    barrier(CLK_LOCAL_MEM_FENCE);

    if((localX==0)&&(localY==0))
        atomic_add(keypointCount, workGroupCount);
}

__kernel void consolidateKeypoints(__global ExtremaMap *extremaMapBuffer, int mapIndex, __global EvolutionInfo *evolutionBuffer, int workOffset, __global Keypoint *keypoints, int maxKeypoints, __global int *count)
{
    const int globalX=get_global_id(0);
    const int globalY=get_global_id(1);

    if((mapIndex == 0) && (globalX == 0) && (globalY == 0))
        *count=0;

    barrier(CLK_LOCAL_MEM_FENCE);

    const int inputX=globalX+workOffset;
    const int inputY=globalY+workOffset;

    __global EvolutionInfo *evolution=&evolutionBuffer[mapIndex];

    const int width=evolution->width;

    int pos=evolution->offset+(inputY*width)+inputX;
    __global ExtremaMap *extremaMap=&extremaMapBuffer[pos];

    const int classId=extremaMap->class_id;

    if(classId < 0)
        return;

    int index=atomic_inc(count);

    if(index>=maxKeypoints)
        return;

    const int octave=evolution->octave;
    float scale=(float)(1<<octave);

    __global Keypoint *keypoint=&keypoints[index];

    keypoint->ptX=inputX*scale;
    keypoint->ptY=inputY*scale;
    keypoint->size=evolution->pointSize;
    keypoint->angle=0.0;
    keypoint->response=extremaMap->response;
    keypoint->octave=evolution->octave;
    keypoint->class_id=classId;

#ifdef TRACK_REMOVED
    if(classId >= 16)
    { 
        if(classId >= 32)
        {
            keypoint->class_id=(classId-32);
            keypoint->removed=2;
        }
        else
        {
            keypoint->class_id=(classId-16);
            keypoint->removed=1;
        }
    }
    else
    {
        keypoint->class_id=classId;
        keypoint->removed=0;
    }
#endif //TRACK_REMOVED
}

float2 calculateSubPixelBuffer(__global float *input, int offset, int width, int height, float scale, int x, int y)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;

    float2 delta;

    int evoPosX=x/scale;
    int evoPosY=y/scale;

    int pos=offset+(evoPosY*width)+evoPosX;

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
    float dx=0.5f*(valueX1Y-valueX_1Y);
    float dy=0.5f*(valueXY1-valueXY_1);

    // Compute the Hessian
    float dxx=valueX1Y+valueX_1Y-2.0f*value;
    float dyy=valueXY1+valueXY_1-2.0f*value;
    float dxy=(0.25f*(valueX1Y1+valueX_1Y_1))-(0.25f*(valueX1Y_1+valueX_1Y1));

    //solve Ax=B
    float det=(dxx*dyy)-(dxy*dxy);
    float invDet=0.0f;

    if(det!=0.0)
        invDet=1.0f/det;

    delta.x=invDet*(dxy*dy-dyy*dx);
    delta.y=invDet*(dxy*dx-dxx*dy);

    return delta;
}

__kernel void subPixelRefinement(__global float *det, __global EvolutionInfo *evolutionBuffer, __global Keypoint *keypoints, __global Keypoint *filteredKeypoints, __global int *count)
{
    int index=get_global_id(0);

    if(index == 0)
        (*count)=0;

    barrier(CLK_LOCAL_MEM_FENCE);

    __global Keypoint *keypoint=&keypoints[index];
    __global EvolutionInfo *evolution=&evolutionBuffer[keypoint->class_id];

    float scale=(float)(1<<evolution->octave);
    float2 delta=calculateSubPixelBuffer(det, evolution->offset, evolution->width, evolution->height, scale, keypoint->ptX, keypoint->ptY);

#ifndef TRACK_REMOVED
    if((delta.x<-1.0f)||(delta.x>1.0f) || (delta.y < -1.0f)||(delta.y > 1.0f))
    {
        return;
    }
    else
#endif //TRACK_REMOVED
    {
        int filteredIndex=atomic_inc(count);

        __global Keypoint *filteredKeypoint=&filteredKeypoints[filteredIndex];

        filteredKeypoint->class_id=keypoint->class_id;
        filteredKeypoint->octave=keypoint->octave;
        filteredKeypoint->size=keypoint->size*2.0f;
//        filteredKeypoint->ptX=keypoint->ptX+(delta.x*scale)+0.5f*(scale-1.0f);
//        filteredKeypoint->ptY=keypoint->ptY+(delta.y*scale)+0.5f*(scale-1.0f);
        filteredKeypoint->ptX=keypoint->ptX+(delta.x*scale);
        filteredKeypoint->ptY=keypoint->ptY+(delta.y*scale);

        filteredKeypoint->response=keypoint->response;
        filteredKeypoint->angle=keypoint->angle;

#ifdef TRACK_REMOVED
        if((delta.x<-1.0f)||(delta.x>1.0f)||(delta.y<-1.0f)||(delta.y>1.0f))
            filteredKeypoint->removed=3;
        else
            filteredKeypoint->removed=keypoint->removed;
#endif //TRACK_REMOVED
    }
}

inline void atomicMaxLocal(volatile __local float *addr, float value)
{
    union
    {
        unsigned int u32;
        float f32;
    } next, expected, current;

    current.f32=*addr;

    do
    {
        expected.f32=current.f32;

        if(value<expected.f32)
            break;

        next.f32=value;
        current.u32=atomic_cmpxchg((volatile __local unsigned int *)addr, expected.u32, next.u32);
    } while(current.u32!=expected.u32);

}

inline void atomicMaxGlobal(volatile __global float *addr, float value)
{
    union
    {
        unsigned int u32;
        float f32;
    } next, expected, current;

    current.f32=*addr;

    do
    {
        expected.f32=current.f32;

        if(value<expected.f32)
            break;

        next.f32=value;
        current.u32=atomic_cmpxchg((volatile __global unsigned int *)addr, expected.u32, next.u32);
    } while(current.u32!=expected.u32);

}

__constant float gauss25[7][7]=
{
    {0.02546481f, 0.02350698f, 0.01849125f, 0.01239505f, 0.00708017f, 0.00344629f, 0.00142946f},
    {0.02350698f, 0.02169968f, 0.01706957f, 0.01144208f, 0.00653582f, 0.00318132f, 0.00131956f},
    {0.01849125f, 0.01706957f, 0.01342740f, 0.00900066f, 0.00514126f, 0.00250252f, 0.00103800f},
    {0.01239505f, 0.01144208f, 0.00900066f, 0.00603332f, 0.00344629f, 0.00167749f, 0.00069579f},
    {0.00708017f, 0.00653582f, 0.00514126f, 0.00344629f, 0.00196855f, 0.00095820f, 0.00039744f},
    {0.00344629f, 0.00318132f, 0.00250252f, 0.00167749f, 0.00095820f, 0.00046640f, 0.00019346f},
    {0.00142946f, 0.00131956f, 0.00103800f, 0.00069579f, 0.00039744f, 0.00019346f, 0.00008024f}
};
__constant int id[13]={6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6};
__constant int2 points[109]=
{
                      {-3, -5},{-2, -5},{-1, -5},{0, -5},{1, -5},{2, -5},{3, -5},
             {-4, -4},{-3, -4},{-2, -4},{-1, -4},{0, -4},{1, -4},{2, -4},{3, -4},{4, -4},
    {-5, -3},{-4, -3},{-3, -3},{-2, -3},{-1, -3},{0, -3},{1, -3},{2, -3},{3, -3},{4, -3},{5, -3},
    {-5, -2},{-4, -2},{-3, -2},{-2, -2},{-1, -2},{0, -2},{1, -2},{2, -2},{3, -2},{4, -2},{5, -2},
    {-5, -1},{-4, -1},{-3, -1},{-2, -1},{-1, -1},{0, -1},{1, -1},{2, -1},{3, -1},{4, -1},{5, -1},
    {-5,  0},{-4,  0},{-3,  0},{-2,  0},{-1,  0},{0,  0},{1,  0},{2,  0},{3,  0},{4,  0},{5,  0},
    {-5,  1},{-4,  1},{-3,  1},{-2,  1},{-1,  1},{0,  1},{1,  1},{2,  1},{3,  1},{4,  1},{5,  1},
    {-5,  2},{-4,  2},{-3,  2},{-2,  2},{-1,  2},{0,  2},{1,  2},{2,  2},{3,  2},{4,  2},{5,  2},
    {-5,  3},{-4,  3},{-3,  3},{-2,  3},{-1,  3},{0,  3},{1,  3},{2,  3},{3,  3},{4,  3},{5,  3},
             {-4,  4},{-3,  4},{-2,  4},{-1,  4},{0,  4},{1,  4},{2,  4},{3,  4},{4,  4},
                      {-3,  5},{-2,  5},{-1,  5},{0,  5},{1,  5},{2,  5},{3,  5}
};
//    __constant int2 points[109]=
//    {
//                          {-5, -3},{-5, -2},{-5, -1},{-5, 0},{-5, 1},{-5, 2},{-5, 3},
//                 {-4, -4},{-4, -3},{-4, -2},{-4, -1},{-4, 0},{-4, 1},{-4, 2},{-4, 3},{-4, 4},
//        {-3, -5},{-3, -4},{-3, -3},{-3, -2},{-3, -1},{-3, 0},{-3, 1},{-3, 2},{-3, 3},{-3, 4},{-3, 5},
//        {-2, -5},{-2, -4},{-2, -3},{-2, -2},{-2, -1},{-2, 0},{-2, 1},{-2, 2},{-2, 3},{-2, 4},{-2, 5},
//        {-1, -5},{-1, -4},{-1, -3},{-1, -2},{-1, -1},{-1, 0},{-1, 1},{-1, 2},{-1, 3},{-1, 4},{-1, 5},
//        { 0, -5},{ 0, -4},{ 0, -3},{ 0, -2},{ 0, -1},{ 0, 0},{ 0, 1},{ 0, 2},{ 0, 3},{ 0, 4},{ 0, 5},
//        { 1, -5},{ 1, -4},{ 1, -3},{ 1, -2},{ 1, -1},{ 1, 0},{ 1, 1},{ 1, 2},{ 1, 3},{ 1, 4},{ 1, 5},
//        { 2, -5},{ 2, -4},{ 2, -3},{ 2, -2},{ 2, -1},{ 2, 0},{ 2, 1},{ 2, 2},{ 2, 3},{ 2, 4},{ 2, 5},
//        { 3, -5},{ 3, -4},{ 3, -3},{ 3, -2},{ 3, -1},{ 3, 0},{ 3, 1},{ 3, 2},{ 3, 3},{ 3, 4},{ 3, 5},
//                 { 4, -4},{ 4, -3},{ 4, -2},{ 4, -1},{ 4, 0},{ 4, 1},{ 4, 2},{ 4, 3},{ 4, 4},
//                          { 5, -3},{ 5, -2},{ 5, -1},{ 5, 0},{ 5, 1},{ 5, 2},{ 5, 3}
//    };

//__kernel void computeOrientation(__global Keypoint *keypoints, __global float *dx, __global float *dy, __global EvolutionInfo *evolution, __local float *localMem)
__kernel void computeOrientation(__global Keypoint *keypoints, __global float *dx, __global float *dy, __global EvolutionInfo *evolution, __global float *dataBuffer)
{
//    __local float *resX=localMem;
//    __local float *resY=&localMem[109];
//    __local float *angles=&localMem[109*2];
//    __local float *maxLength=&localMem[109*3];
    __local float resX[109];
    __local float resY[109];
    __local float angles[109];
    __local float maxLength;

    const int index=get_global_id(0);
    const int groupIndex=get_global_id(1);

    const int dataPos=index*((109*3)+(42*3));
    const int anglePos=dataPos+(109*3)+(groupIndex*3);

    __global Keypoint *keypoint=&keypoints[index];
    const float twoPi=2.0*M_PI;

    //reset maxLength value
    if(groupIndex==0)
        maxLength=0.0f;

    // Get the information from the keypoint
    int level=keypoint->class_id;
    float ratio=(float)(1<<keypoint->octave);
    int scale=round(0.5 * keypoint->size/ratio);
    float xf=keypoint->ptX/ratio;
    float yf=keypoint->ptY/ratio;

    int startIndex=groupIndex*3;
    int endIndex=startIndex+3;

    if(endIndex>109)
        endIndex=109;
    
    int offset=evolution[level].offset;
    int width=evolution[level].width;

    // Calculate derivatives responses for points within radius of 3*scale
    //calculating the info for local work group, performing 42 threads so calculating
    //3 points per thread
    for(int i=startIndex; i<endIndex; i++)
    {
        int x=points[i].x;
        int y=points[i].y;

        int ix=round(xf+(x*scale));
        int iy=round(yf+(y*scale));
        
        float gweight=gauss25[id[x+6]][id[y+6]];
        float valueX=gweight*dx[offset+iy*width+ix];
        float valueY=gweight*dy[offset+iy*width+ix];
        float angle;

        resX[i]=valueX;
        resY[i]=valueY;
        angle=atan2(valueY, valueX);

        if(angle<0.0)
            angle=angle+twoPi;

        angles[i]=angle;
        dataBuffer[dataPos+(i*3)]=x;
        dataBuffer[dataPos+(i*3)+1]=y;
        dataBuffer[dataPos+(i*3)+2]=angle;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    //calculating angle sums
    //performing this on 42 threads
    float sumX, sumY;
    float length;

    float angleStart=(groupIndex)*0.15f;
    float angleEnd=angleStart+M_PI/3.0f;
    bool zeroCross=false;

    if(angleEnd>twoPi)
    {
        angleEnd=angleEnd-twoPi;
        zeroCross=true;
    }

    sumX=sumY=0.f;

    for(size_t i=0; i<109; ++i)
    {
        float angle=angles[i];
        bool add=false;

        if(zeroCross)
        {
            if((angle>angleStart)||(angle<angleEnd))
                add=true;
        }
        else
        {
            if((angle>angleStart)&&(angle<angleEnd))
                add=true;
        }

        if(add)
        {
            sumX+=resX[i];
            sumY+=resY[i];
        }
    }

    length=(sumX*sumX)+(sumY*sumY);
    atomicMaxLocal(&maxLength, length);
    dataBuffer[anglePos]=angleStart;
    dataBuffer[anglePos+1]=angleEnd;
    dataBuffer[anglePos+2]=length;

    barrier(CLK_LOCAL_MEM_FENCE);

    //check if maxLength was ours, this is a little "not right" as there is the possibilty
    // that more than one point has the same value. But I am going to assume none will ':)

    float maxValue=maxLength;

    if(maxValue == length)
        keypoint->angle=atan2(sumY, sumX);
}

void fillValuesMLDB(float* values, int sample_step, int level, float xf, float yf, float co, float si, float scale,
    __global float *imageBuffer, __global float *dxBuffer, __global float *dyBuffer, int width, int height, int channels, int patternSize)
{

    int pattern_size=patternSize;
    int nr_channels=channels;// options_.descriptor_channels;
    int valpos=0;

    for(int i=-pattern_size; i < pattern_size; i+=sample_step)
    {
        for(int j=-pattern_size; j < pattern_size; j+=sample_step)
        {

            float di=0.0, dx=0.0, dy=0.0;
            int nsamples=0;

            for(int k=i; k < i+sample_step; k++)
            {
                for(int l=j; l < j+sample_step; l++)
                {

                    float sample_y=yf+(l * co * scale+k * si * scale);
                    float sample_x=xf+(-l * si * scale+k * co * scale);

                    int y1=round(sample_y);
                    int x1=round(sample_x);

                    float ri=imageBuffer[(y1*width)+x1];
                    di+=ri;

                    if(nr_channels > 1)
                    {
                        float rx=dxBuffer[(y1*width)+x1];
                        float ry=dyBuffer[(y1*width)+x1];
                        if(nr_channels==2)
                        {
                            dx+=sqrt(rx * rx+ry * ry);
                        }
                        else
                        {
                            float rry=rx * co+ry * si;
                            float rrx=-rx * si+ry * co;
                            dx+=rrx;
                            dy+=rry;
                        }
                    }
                    nsamples++;
                }
            }

            di/=nsamples;
            dx/=nsamples;
            dy/=nsamples;

            values[valpos]=di;

            if(nr_channels > 1) values[valpos+1]=dx;

            if(nr_channels > 2) values[valpos+2]=dy;

            valpos+=nr_channels;
        }
    }
}

void binaryComparisonsMLDB(float *values, __global unsigned char *desc, int count, int *dpos, int channels)
{
#define TOGGLE_FLT(x) ((x) ^ ((int)(x) < 0 ? 0x7fffffff : 0))

    int nr_channels=channels;// options_.descriptor_channels;
    int* ivalues=(int*)values;

    for(int i=0; i < count * nr_channels; i++)
    {
        ivalues[i]=TOGGLE_FLT(ivalues[i]);
    }
#undef TOGGLE_FLT

    int localPos=*dpos;

    for(int pos=0; pos < nr_channels; pos++)
    {
        for(int i=0; i < count; i++)
        {
            int ival=ivalues[nr_channels * i+pos];
            for(int j=i+1; j < count; j++)
            {
                int res=ival > ivalues[nr_channels * j+pos];

                desc[(localPos>>3)]|=(res<<(localPos&7));
                localPos++;
            }
        }
    }

    (*dpos)=localPos;
}

__kernel void getMLDBDescriptor(__global Keypoint *keypoints, __global unsigned char *desc, __global float *imageBuffer, __global float *dxBuffer, __global float *dyBuffer, __global EvolutionInfo *evolution, int patternSize)
{
    const int index=get_global_id(0);
    const int channels=3;
    float values[16*3];
    const float size_mult[3]={1, 2.0f/3.0f, 1.0f/2.0f};

    __global Keypoint *keypoint=&keypoints[index];

    float ratio=(float)(1<<keypoint->octave);
    float scale=(float)round(0.5f * keypoint->size/ratio);
    float xf=keypoint->ptX/ratio;
    float yf=keypoint->ptY/ratio;
    float co=cos(keypoint->angle);
    float si=sin(keypoint->angle);

    int evolutionLevel=keypoint->class_id;
    int offset=evolution[evolutionLevel].offset;
    int width=evolution[evolutionLevel].width;
    int height=evolution[evolutionLevel].height;

    __global float *image=&imageBuffer[offset];
    __global float *dx=&dxBuffer[offset];
    __global float *dy=&dyBuffer[offset];

	const int descriptorSize=ceil((float)((6+36+120)*channels)/8);
	__global unsigned char *descriptor=&desc[index*descriptorSize];

    int dpos=0;
    for(int lvl=0; lvl < 3; lvl++)
    {
        int val_count=(lvl+2) * (lvl+2);
        int sample_step=(int)(ceil(patternSize * size_mult[lvl]));

        fillValuesMLDB(values, sample_step, keypoint->class_id, xf, yf, co, si, scale, image, dx, dy, width, height, channels, patternSize);
        binaryComparisonsMLDB(values, descriptor, val_count, &dpos, channels);
    }
}
