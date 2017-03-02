#include "templates.h"

template <int RADIUS> __global__ void ConvColGPU(struct Conv_t s)
{
    //__global__ void ConvColGPU(float *d_Result, float *d_Data, int width, int
    //pitch, int height) {
    __shared__ float data[CONVCOL_W * (CONVCOL_H+2*RADIUS)];
    const int tx=threadIdx.x;
    const int ty=threadIdx.y;
    const int miny=blockIdx.y * CONVCOL_H;
    const int maxy=min(miny+CONVCOL_H, s.height)-1;
    const int totStart=miny-RADIUS;
    const int totEnd=maxy+RADIUS;
    const int colStart=blockIdx.x * CONVCOL_W+tx;
    const int colEnd=colStart+(s.height-1) * s.pitch;
    const int smemStep=CONVCOL_W * CONVCOL_S;
    const int gmemStep=s.pitch * CONVCOL_S;

    if(colStart < s.width)
    {
        int smemPos=ty * CONVCOL_W+tx;
        int gmemPos=colStart+(totStart+ty) * s.pitch;
        for(int y=totStart+ty; y<=totEnd; y+=blockDim.y)
        {
            if(y < 0)
                data[smemPos]=s.d_Data[colStart];
            else if(y>=s.height)
                data[smemPos]=s.d_Data[colEnd];
            else
                data[smemPos]=s.d_Data[gmemPos];
            smemPos+=smemStep;
            gmemPos+=gmemStep;
        }
    }
    __syncthreads();
    if(colStart < s.width)
    {
        int smemPos=ty * CONVCOL_W+tx;
        int gmemPos=colStart+(miny+ty) * s.pitch;
        for(int y=miny+ty; y<=maxy; y+=blockDim.y)
        {
            float sum=0.0f;
            for(int i=0; i<=2*RADIUS; i++)
                sum+=data[smemPos+i * CONVCOL_W]*d_Kernel[i];
            s.d_Result[gmemPos]=sum;
            smemPos+=smemStep;
            gmemPos+=gmemStep;
        }
    }
}

//double Template1(SeparableFilter, RADIUS)(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, float *h_Kernel)
Template1(separableFilter, RADIUS)(__read_only image2d_t inimg, __write_only image2d_t outimg, float *h_Kernel)
{
    int width=inimg.width;
    int pitch=inimg.pitch;
    int height=inimg.height;
    float *d_DataA=inimg.d_data;
    float *d_DataB=outimg.d_data;
    float *d_Temp=temp.d_data;

//    if(d_DataA==NULL||d_DataB==NULL||d_Temp==NULL)
//    {
//        printf("SeparableFilter: missing data\n");
//        return 0.0;
//    }
    // TimerGPU timer0(0);
    const unsigned int kernelSize=(2*RADIUS+1)*sizeof(float);
//    safeCall(cudaMemcpyToSymbolAsync(d_Kernel, h_Kernel, kernelSize));

    dim3 blockGridRows(iDivUp(width, CONVROW_W), height);
    dim3 threadBlockRows(CONVROW_W+2*RADIUS);
    struct Conv_t s;
    s.d_Result=d_Temp;
    s.d_Data=d_DataA;
    s.width=width;
    s.pitch=pitch;
    s.height=height;
    ConvRowGPU<RADIUS><<<blockGridRows, threadBlockRows>>> (s);
    // checkMsg("ConvRowGPU() execution failed\n");
    // safeCall(cudaThreadSynchronize());
    dim3 blockGridColumns(iDivUp(width, CONVCOL_W), iDivUp(height, CONVCOL_H));
    dim3 threadBlockColumns(CONVCOL_W, CONVCOL_S);
    s.d_Result=d_DataB;
    s.d_Data=d_Temp;
    ConvColGPU<RADIUS><<<blockGridColumns, threadBlockColumns>>> (s);
    // checkMsg("ConvColGPU() execution failed\n");
    // safeCall(cudaThreadSynchronize());

    double gpuTime=0;  // timer0.read();
#ifdef VERBOSE
    printf("SeparableFilter time =        %.2f ms\n", gpuTime);
#endif
    return gpuTime;
}


//double TEMPLATE1(lowPass, RADIUS)(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var)
__kernel TEMPLATE1(lowPass, RADIUS)(__read_only image2d_t inimg, __write_only image2d_t outimg, double var)
{
    float kernel[2*RADIUS+1];
    float kernelSum=0.0f;

    for(int j=-RADIUS; j<=RADIUS; j++)
    {
        kernel[j+RADIUS]=(float)expf(-(double)j * j/2.0/var);
        kernelSum+=kernel[j+RADIUS];
    }
    
    for(int j=-RADIUS; j<=RADIUS; j++) 
        kernel[j+RADIUS]/=kernelSum;

    separableFilter<RADIUS>(inimg, outimg, kernel);
}