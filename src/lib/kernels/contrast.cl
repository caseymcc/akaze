
__kernel void computeContrast(read_only image2d_t input, int width, int height, write_only image2d_t output)//, 
    //__local float *imageCache, int imageCacheSize)
{
    const sampler_t nearestClampSampler=CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;

    const int globalX=get_global_id(0);
    const int globalY=get_global_id(1);
    const int localX=get_local_id(0);
    const int localY=get_local_id(1);
    const int localXSize=get_local_size(0);
    const int localYSize=get_local_size(1);

    const int kernelSize=3;
    const int halfKernelSize=1;
    int cacheIndex=((localY*localXSize)+localX)*2;

    __local float imageCache[18*18];
    int imageCacheX=localXSize+2;
    int imageCacheY=localYSize+2;

    int imageX=globalX-halfKernelSize;
    int imageY=globalY-halfKernelSize;

    if(imageX+imageCacheX>width+1)
        imageCacheX=width-imageX+1;
    if(imageY+imageCacheY>height+1)
        imageCacheY=height-imageY+1;

    const int imageCacheSize=imageCacheX*imageCacheY;

//build up image cache
    if(cacheIndex < imageCacheSize)
    {
        const int shiftX=(localX*2);
        imageX+=shiftX;
        imageY+=localY*2;

        if(shiftX>imageCacheX)
        {
            imageX=imageX-imageCacheX;
            imageY++;
        }

        imageCache[cacheIndex]=read_imagef(input, nearestClampSampler, (int2)(imageX, imageY)).x;
        imageCache[cacheIndex+1]=read_imagef(input, nearestClampSampler, (int2)(imageX+1, imageY)).x;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    //used extras for memcopy so now they can be ignored
    if((globalX >= width) || (globalY >= height))
        return;

    const int cacheX=localX+halfKernelSize;
    const int cacheY=localY+halfKernelSize;

    int cacheXKernel=(cacheY*imageCacheX)+cacheX;

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
