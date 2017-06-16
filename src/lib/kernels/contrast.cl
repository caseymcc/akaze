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
            imageCache[cacheIndex+i]=read_imagef(input, nearestClampSampler, (int2)(indexX, indexY)).x;
            indexX++;
        }
    }

    (*imageCacheX)=imgCacheX;
    (*imageCacheY)=imgCacheY;

    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void computeContrast(read_only image2d_t input, int width, int height, write_only image2d_t output)//, 
    //__local float *imageCache, int imageCacheSize)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;

    const int globalX=get_global_id(0);
    const int globalY=get_global_id(1);
    const int groupX=get_group_id(0);
    const int groupY=get_group_id(1);
    const int localX=get_local_id(0);
    const int localY=get_local_id(1);
    const int localXSize=get_local_size(0);
    const int localYSize=get_local_size(1);

//    const int kernelSize=3;
    const int halfKernelSize=1;
    int cacheIndex=((localY*localXSize)+localX)*2;

    __local float imageCache[18*18];
    int imageCacheX=localXSize+2;
    int imageCacheY=localYSize+2;

    int imageX=(groupX*localXSize)-halfKernelSize;
    int imageY=(groupY*localYSize)-halfKernelSize;

    if(imageX+imageCacheX>width+1)
        imageCacheX=width-imageX+1;
    if(imageY+imageCacheY>height+1)
        imageCacheY=height-imageY+1;

    const int imageCacheSize=imageCacheX*imageCacheY;

//build up image cache
    if(cacheIndex < imageCacheSize)
    {
//        const int shiftX=(localX*2);
//        imageX+=shiftX;
//        imageY+=localY*2;
//
//        if(shiftX>imageCacheX)
//        {
//            imageX=imageX-imageCacheX;
//            imageY++;
//        }
        const int indexY=cacheIndex/imageCacheX;
        imageY+=indexY;
        imageX+=cacheIndex-(indexY*imageCacheX);

        float value1=read_imagef(input, nearestClampSampler, (int2)(imageX, imageY)).x;
        float value2=read_imagef(input, nearestClampSampler, (int2)(imageX+1, imageY)).x;

        imageCache[cacheIndex]=value1;
        imageCache[cacheIndex+1]=value2;
//        imageCache[cacheIndex]=read_imagef(input, nearestClampSampler, (int2)(imageX, imageY)).x;
//        imageCache[cacheIndex+1]=read_imagef(input, nearestClampSampler, (int2)(imageX+1, imageY)).x;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    //used extras for memcopy so now they can be ignored
    if((globalX >= width) || (globalY >= height))
        return;

    int cacheXKernel=(localY*imageCacheX)+localX;

//calculate first order derivatives
    const float ul=imageCache[cacheXKernel];
    const float ur=imageCache[cacheXKernel+2];
    const float ll=imageCache[cacheXKernel+(2*imageCacheX)];
    const float lr=imageCache[cacheXKernel+(2*imageCacheX)+2];

    float dx=3*(ur+lr-ul-ll)+10*(imageCache[cacheXKernel+imageCacheX+2]-imageCache[cacheXKernel+imageCacheX]);
    float dy=3*(ll+lr-ul-ur)+10*(imageCache[cacheXKernel+(2*imageCacheX)+1]-imageCache[cacheXKernel+1]);

//calculate magnitude
    const float value=sqrt(dx*dx+dy*dy);

    write_imagef(output, (int2)(globalX, globalY), value);
}

__kernel void computeFlow(read_only image2d_t input, int width, int height, write_only image2d_t output, int diffusivity, int index, __global float *contrast)//, __local float *imageCache)
{
    const int globalX=get_global_id(0);
    const int globalY=get_global_id(1);
    const int localX=get_local_id(0);
    const int localY=get_local_id(1);
    
    __local float imageCache[18*18];
    int imageCacheX;
    int imageCacheY;

    copyToLocal(input, width, height, 1, localX, localY, imageCache, &imageCacheX, &imageCacheY);

    if((globalX>=width)||(globalY>=height))
        return;

    const int cachePos=(localY*imageCacheX)+localX;

    const float ul=imageCache[cachePos];
    const float ur=imageCache[cachePos+2];
    const float ll=imageCache[cachePos+(2*imageCacheX)];
    const float lr=imageCache[cachePos+(2*imageCacheX)+2];

    const float dx=3*(ur+lr-ul-ll)+10*(imageCache[cachePos+imageCacheX+2]-imageCache[cachePos+imageCacheX]);
    const float dy=3*(ll+lr-ul-ur)+10*(imageCache[cachePos+(2*imageCacheX)+1]-imageCache[cachePos+1]);
    float value;

    const float k=contrast[0]*pow(0.75, index);
    const float inv_k=1.0/(k * k);

    switch(diffusivity)
    {
    case 0: //pmG1
        value=exp(-inv_k*(dx*dx+dy*dy));
        break;
    case 1: //pmG2
        value=1.0/(1.0+inv_k*(dx*dx+dy*dy));
        break;
    case 2: //weickert
        {
            const float diff=inv_k*(dx*dx+dy*dy);
            value=1.0-exp(-3.315/(diff*diff*diff*diff));
        }
        break;
    case 3: //charbonnier
        value=1.0/sqrt(1.0+inv_k*(dx*dx+dy*dy));
        break;
    }

    write_imagef(output, (int2)(globalX, globalY), value);
}
