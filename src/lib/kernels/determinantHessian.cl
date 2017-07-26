void copyToLocal(read_only image2d_t input, int width, int height, int border, int localX, int localY, __local float *imageCache, int *imageCacheX, int *imageCacheY)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;

    const int groupX=get_group_id(0);
    const int groupY=get_group_id(1);
    const int localSizeX=get_local_size(0);
    const int localSizeY=get_local_size(1);

    const int imageX=(groupX*localSizeX)-border;
    const int imageY=(groupY*localSizeY)-border;

    int imgCacheX=localSizeX+(2*border);
    int imgCacheY=localSizeY+(2*border);

    if(imageX+imgCacheX>=width)
        imgCacheX=width-imageX+1;
    if(imageY+imgCacheY>=height)
        imgCacheY=height-imageY+1;

    const int imageCacheSize=imgCacheX*imgCacheY;
    const int perItemCache=ceil((float)imageCacheSize/(localSizeX*localSizeY));
    const int cacheIndex=((localY*localSizeX)+localX)*perItemCache;

    if(cacheIndex < imageCacheSize)
    {
        const int cacheY=cacheIndex/imgCacheX;
        const int indexY=imageY+cacheY;
        int indexX=imageX+cacheIndex-(cacheY*imgCacheX);

        for(int i=0; i<perItemCache; ++i)
        {
            float value=read_imagef(input, nearestClampSampler, (int2)(indexX, indexY)).x;
            imageCache[cacheIndex+i]=value;
            indexX++;
        }
    }

    (*imageCacheX)=imgCacheX;
    (*imageCacheY)=imgCacheY;
}

__kernel void determinantHessian(read_only image2d_t input, int width, int height, __constant float *edgeKernel, __constant float *smoothKernel, int kernelSize, float scale, write_only image2d_t dx, 
    write_only image2d_t dy, write_only image2d_t det, __local float *imageCache)
{
    const int globalX=get_global_id(0);
    const int globalY=get_global_id(1);
    const int localX=get_local_id(0);
    const int localY=get_local_id(1);
    const int halfKernelSize=kernelSize/2;

    bool process=true;
    int imageCacheX;
    int imageCacheY;

    //copy over image to local cache.
    copyToLocal(input, width, height, halfKernelSize*2, localX, localY, imageCache, &imageCacheX, &imageCacheY);

    //copy needs to finish before we read
    barrier(CLK_LOCAL_MEM_FENCE);

    if((globalX<width)&&(globalY<height))
        process=false;

    if(process)
    {
        int cacheStride=imageCacheX*imageCacheY;

        //perform convolve for dx and dy on y axis
        const int posY=localY*2;
        float2 sumDx=0.0f;
        float2 sumDy=0.0f;

        if(posY<imageCacheY)
        {
            int cachePos=(posY*imageCacheX)+localX;

            for(int i=0; i<kernelSize; i++)
            {
                float2 value=(float2)(imageCache[cachePos], imageCache[cachePos+imageCacheX]);

                sumDx+=smoothKernel[i]*value;
                sumDy+=edgeKernel[i]*value;
                cachePos++;
            }
        }
    }

    //overwriting local memory so need to make sure we are done before starting
    barrier(CLK_LOCAL_MEM_FENCE);

    if(process)
    {
        //replace original image in cache with dx and fill in next part of cache with dy
        int cachePos=((posY)*imageCacheX)+localX+halfKernelSize;
        int cachePos2=((posY)*imageCacheX)+localX+halfKernelSize+cacheStride;

        if(posY<imageCacheY)
        {
            imageCache[cachePos]=sumDx.x;
            imageCache[cachePos+imageCacheX]=sumDx.y;

            imageCache[cachePos2]=sumDy.x;
            imageCache[cachePos2+imageCacheX]=sumDy.y;
        }
    }

    //about to read from modified local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    if(process)
    {
        //perform convolve for dx and dy on X axis
        const int posX=localX*2;
        sumDx=0.0f;
        sumDy=0.0f;

        if(posX<imageCacheX)
        {
            cachePos=(localY*imageCacheX)+posX+halfKernelSize;
            cachePos2=(localY*imageCacheX)+posX+halfKernelSize+cacheStride;

            for(int i=0; i<kernelSize; i++)
            {
                float2 value1=(float2)(imageCache[cachePos], imageCache[cachePos+1]);
                float2 value2=(float2)(imageCache[cachePos2], imageCache[cachePos2+1]);

                sumDx+=edgeKernel[i]*value1;
                sumDy+=smoothKernel[i]*value2;

                cachePos+=2;
                cachePos2+=2;
            }
            sumDx=sumDx*scale;
            sumDy=sumDy*scale;
        }
    }

    //overwriting local memory again
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(process)
    {
        //update cache with calculated values and write back dx and dy
        cachePos=((localY)*imageCacheX)+posX+halfKernelSize;
        cachePos2=((localY)*imageCacheX)+posX+halfKernelSize+cacheStride;

        if(posX<imageCacheX)
        {
            imageCache[cachePos]=sumDx.x;
            imageCache[cachePos+1]=sumDx.y;

            //write back dx image for later use
            write_imagef(dx, (int2)(globalX, globalY), sumDx.x);
            write_imagef(dx, (int2)(globalX+1, globalY), sumDx.y);

            imageCache[cachePos2]=sumDy.x;
            imageCache[cachePos2+1]=sumDy.y;

            //write back dy image for later use
            write_imagef(dy, (int2)(globalX, globalY), sumDx.x);
            write_imagef(dy, (int2)(globalX+1, globalY), sumDx.y);
        }
    }
    
    //local memory modified again, we are not using the info from the image writes so just a CLK_LOCAL_MEM_FENCE here
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(process)
    {
        //now perform partial derivates dxx dxy dyy using same kernel, cache currently holds dx and dy
    }

    write_imagef(output, (int2)(globalX, globalY), sum);
}