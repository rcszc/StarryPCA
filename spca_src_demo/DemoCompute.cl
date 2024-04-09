// kernel function.
__kernel void DemoMatrixCalc(
    __global const float* MatrixIn, __global const float* ConvKernelA, __global const float* ConvKernelB, 
    __global const float* ConvParam, __global float* MatrixOut
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int width = get_global_size(0);
    int height = get_global_size(1);

    int KernelSizeX = (int)ConvParam[0];
    int KernelSizeY = (int)ConvParam[1];

    int KernelCenterX = KernelSizeX / 2;
    int KernelCenterY = KernelSizeY / 2;

    float ResultValue = 0.0f;

    for (int kx = 0; kx < KernelSizeX; ++kx) {
        for (int ky = 0; ky < KernelSizeY; ++ky) {

            int InputX = i + kx - KernelCenterX;
            int InputY = j + ky - KernelCenterY;

            if (InputX >= 0 && InputX < width && InputY >= 0 && InputY < height) {

                float InputVal = MatrixIn[InputY * width + InputX];
                float KernelValA = ConvKernelA[ky * KernelSizeX + kx];
                float KernelValB = ConvKernelB[ky * KernelSizeX + kx];

                ResultValue += InputVal * KernelValA + InputVal * KernelValB;
            }
        }
    }

    MatrixOut[j * width + i] = ResultValue;
}