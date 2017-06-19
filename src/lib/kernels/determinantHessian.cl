
__kernel void determinantHessian(read_only image2d_t input, int width, int height, __constant float *kernelX, __constant float *kernelY, int kernelSize, float scale, write_only image2d_t output, __local float *imageCache)
{
    const int globalX=get_global_id(0);
    const int globalY=get_global_id(1);
    const int localX=get_local_id(0);
    const int localY=get_local_id(1);
    const int halfKernelSize=kernelSize/2;

    int imageCacheX;
    int imageCacheY;

    copyToLocal(input, width, height, halfKernelSize, localX, localY, imageCache, &imageCacheX, &imageCacheY);

    if((globalX>=width)||(globalY>=height))
        return;

    //perform convolve on X
    const int posY=localY*2;
    float2 sum2=0.0f;

    if(posY<imageCacheY)
    {
        int cachePos=(posY*imageCacheX)+localX;

        for(int i=0; i<kernelSize; i++)
        {
            float2 value=(float2)(imageCache[cachePos], imageCache[cachePos+imageCacheX]);

            sum2+=kernelX[i]*value;
            cachePos++;
        }
    }

    int cachePos=((posY)*imageCacheX)+localX+halfKernelSize;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(posY<imageCacheY)
    {
        imageCache[cachePos]=sum2.x;
        imageCache[cachePos+imageCacheX]=sum2.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //    cachePos=((localY+halfKernelSize)*imageCacheX)+localX+halfKernelSize;
    //    write_imagef(output, (int2)(globalX, globalY), imageCache[cachePos]);

    //perform convolve on Y
    cachePos=(localY*imageCacheX)+localX+halfKernelSize;
    float sum=0.0f;

    for(int i=0; i<kernelSize; i++)
    {
        float value=imageCache[cachePos];

        sum+=kernelY[i]*value;
        cachePos+=imageCacheX;
    }

    sum=sum*scale;
    write_imagef(output, (int2)(globalX, globalY), sum);
}