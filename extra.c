// Notes: first private for omp
#include <emmintrin.h>
#include <omp.h>

#define ALIGNSIZE 100
#define PADDEDMATRIXSIZE 1597700

int conv2D(float* in, float* out, int data_size_X, int data_size_Y, float* kernel, int kernel_x, int kernel_y) {
	int kern_cent_X = kernel_x >> 1; // the x coordinate of the kernel's center
	int kern_cent_Y = kernel_y >> 1; // the y coordinate of the kernel's center
  
	int data_size_X2 = data_size_X + (ALIGNSIZE - (data_size_X % ALIGNSIZE)) % ALIGNSIZE; // make sure x is a multiple of 4
	int padded_data_size_X = data_size_X2 + 2 * kern_cent_X;
	int padded_data_size_Y = data_size_Y + 2 * kern_cent_Y;
  
	// Create matrix of 0s for padded input/output
	static float padded_out[PADDEDMATRIXSIZE]__attribute__((aligned(16))); // 1204 * 1204
  
	//lets flip the kernel
	int length_kernel = kernel_x * kernel_y;
	float flip_kernel[length_kernel];
	int len_kernel_MINUS_one = length_kernel - 1;
	for (int x = 0; x < length_kernel; x++) {
		flip_kernel[x] = kernel[len_kernel_MINUS_one - x];
	}
	
	/*****************************/
	/*        400                */
	/*****************************/
  if (data_size_X == 400) {
    static float padded_in[PADDEDMATRIXSIZE]__attribute__((aligned(16))); // 1204 * 1204  
    static int trials;
    trials++;
    
    if (trials > 1) {     
      // fill the preinput matrix with zeros
      int padded_size = (padded_data_size_X)* (padded_data_size_Y); // maybe remove or lower the padded_size bounds
      int padded_size_block = padded_size/100*100;
#pragma omp parallel for
      for (int z = 400; z < padded_size_block; z += 100) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 4 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 8 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 12 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 16 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 20 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 24 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 28 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 32 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 36 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 40 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 44 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 48 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 52 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 56 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 60 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 64 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 68 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 72 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 76 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 80 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 84 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 88 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 92 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 96 + z, _mm_setzero_ps());
      }
      for (int z = padded_size_block; z < padded_size; z += 4) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
      }
    }
		
		// Insert input matrix into padded matrix
		int x_boundary = data_size_X/32*32;
#pragma omp parallel for
		for (int y = 0; y < data_size_Y; y++) {
			float *in_add_offset_orig = in + y * data_size_X;
			float *padded_in_offset_3_orig = padded_in + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
			for (int x = 0; x < x_boundary; x += 32) {
				float *in_add_offset = in_add_offset_orig + x;
				float *padded_in_offset_3 = padded_in_offset_3_orig + x;
				_mm_storeu_ps(padded_in_offset_3, _mm_loadu_ps(in_add_offset));
				_mm_storeu_ps(padded_in_offset_3 + 4, _mm_loadu_ps(in_add_offset + 4));
				_mm_storeu_ps(padded_in_offset_3 + 8, _mm_loadu_ps(in_add_offset + 8));
				_mm_storeu_ps(padded_in_offset_3 + 12, _mm_loadu_ps(in_add_offset + 12));
				_mm_storeu_ps(padded_in_offset_3 + 16, _mm_loadu_ps(in_add_offset + 16));
				_mm_storeu_ps(padded_in_offset_3 + 20, _mm_loadu_ps(in_add_offset + 20));
				_mm_storeu_ps(padded_in_offset_3 + 24, _mm_loadu_ps(in_add_offset + 24));
				_mm_storeu_ps(padded_in_offset_3 + 28, _mm_loadu_ps(in_add_offset + 28));
			}
			for (int x = x_boundary; x < data_size_X; x++) {
				padded_in[(x + kern_cent_X) + (y + kern_cent_Y) * padded_data_size_X] = in[x + y * data_size_X];
			}
		}
    
		// main convolution loop
#pragma omp parallel for
		for(int y = 0; y < data_size_Y; y++) { // the y coordinate of theoutput location we're focusing on
      // because we know we have a minimum size of 400
      for(int x = 0; x < 400; x += 100) { // the x coordinate of the output location we're focusing on
        __m128 sum_vec = _mm_setzero_ps();
        __m128 sum_vec2 = _mm_setzero_ps();
        __m128 sum_vec3 = _mm_setzero_ps();
        __m128 sum_vec4 = _mm_setzero_ps();
        __m128 sum_vec5 = _mm_setzero_ps();
        __m128 sum_vec6 = _mm_setzero_ps();
        __m128 sum_vec7 = _mm_setzero_ps();
        __m128 sum_vec8 = _mm_setzero_ps();
        __m128 sum_vec9 = _mm_setzero_ps();
        __m128 sum_vec10 = _mm_setzero_ps();
        __m128 sum_vec11 = _mm_setzero_ps();
        __m128 sum_vec12 = _mm_setzero_ps();
        __m128 sum_vec13 = _mm_setzero_ps();
        __m128 sum_vec14 = _mm_setzero_ps();
        __m128 sum_vec15 = _mm_setzero_ps();
        __m128 sum_vec16 = _mm_setzero_ps();
        __m128 sum_vec17 = _mm_setzero_ps();
        __m128 sum_vec18 = _mm_setzero_ps();
        __m128 sum_vec19 = _mm_setzero_ps();
        __m128 sum_vec20 = _mm_setzero_ps();
        __m128 sum_vec21 = _mm_setzero_ps();
        __m128 sum_vec22 = _mm_setzero_ps();
        __m128 sum_vec23 = _mm_setzero_ps();
        __m128 sum_vec24 = _mm_setzero_ps();
        __m128 sum_vec25 = _mm_setzero_ps();
        
        for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
          for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
            
            float *padded_in_Offset = padded_in + x + i + (y + j) *padded_data_size_X;
            __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
            
            __m128 input_vec = _mm_loadu_ps(padded_in_Offset);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
            __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset + 4);
            sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
            __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset + 8);
            sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
            __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset + 12);
            sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
            __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset + 16);
            sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
            __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset + 20);
            sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
            __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset + 24);
            sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
            __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset + 28);
            sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
            __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset + 32);
            sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
            __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset + 36);
            sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
            __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset + 40);
            sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
            __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset + 44);
            sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
            __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset + 48);
            sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
            __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset + 52);
            sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
            __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset + 56);
            sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
            __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset + 60);
            sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
            __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset + 64);
            sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
            __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset + 68);
            sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
            __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset + 72);
            sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
            __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset + 76);
            sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
            __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset + 80);
            sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
            __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset + 84);
            sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
            __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset + 88);
            sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
            __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset + 92);
            sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
            __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset + 96);
            sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
          }
        }
        float *outOffset = out + x + y * data_size_X;
        _mm_storeu_ps(outOffset, sum_vec);
        _mm_storeu_ps(outOffset + 4, sum_vec2);
        _mm_storeu_ps(outOffset + 8, sum_vec3);
        _mm_storeu_ps(outOffset + 12, sum_vec4);
        _mm_storeu_ps(outOffset + 16, sum_vec5);
        _mm_storeu_ps(outOffset + 20, sum_vec6);
        _mm_storeu_ps(outOffset + 24, sum_vec7);
        _mm_storeu_ps(outOffset + 28, sum_vec8);
        _mm_storeu_ps(outOffset + 32, sum_vec9);
        _mm_storeu_ps(outOffset + 36, sum_vec10);
        _mm_storeu_ps(outOffset + 40, sum_vec11);
        _mm_storeu_ps(outOffset + 44, sum_vec12);
        _mm_storeu_ps(outOffset + 48, sum_vec13);
        _mm_storeu_ps(outOffset + 52, sum_vec14);
        _mm_storeu_ps(outOffset + 56, sum_vec15);
        _mm_storeu_ps(outOffset + 60, sum_vec16);
        _mm_storeu_ps(outOffset + 64, sum_vec17);
        _mm_storeu_ps(outOffset + 68, sum_vec18);
        _mm_storeu_ps(outOffset + 72, sum_vec19);
        _mm_storeu_ps(outOffset + 76, sum_vec20);
        _mm_storeu_ps(outOffset + 80, sum_vec21);
        _mm_storeu_ps(outOffset + 84, sum_vec22);
        _mm_storeu_ps(outOffset + 88, sum_vec23);
        _mm_storeu_ps(outOffset + 92, sum_vec24);
        _mm_storeu_ps(outOffset + 96, sum_vec25);
      }
    }
    
    /*****************************/
    /*    Greater than 1100      */
    /*****************************/
	} else if (data_size_X > 1100) {   
    static float padded_in[PADDEDMATRIXSIZE]__attribute__((aligned(16))); // 1204 * 1204   
    static int trials;
    trials++;
    
    if (trials > 1) {     
      // fill the preinput matrix with zeros
      int padded_size = (padded_data_size_X)* (padded_data_size_Y); // maybe remove or lower the padded_size bounds
      int padded_size_block = padded_size/100*100;
#pragma omp parallel for
      for (int z = 400; z < padded_size_block; z += 100) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 4 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 8 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 12 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 16 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 20 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 24 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 28 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 32 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 36 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 40 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 44 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 48 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 52 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 56 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 60 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 64 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 68 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 72 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 76 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 80 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 84 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 88 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 92 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 96 + z, _mm_setzero_ps());
      }
      for (int z = padded_size_block; z < padded_size; z += 4) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
      }
    }
	
		// Insert input matrix into padded matrix
		int x_boundary = data_size_X/32*32;
#pragma omp parallel for
		for (int y = 0; y < data_size_Y; y++) {
			float *in_add_offset_orig = in + y * data_size_X;
			float *padded_in_offset_3_orig = padded_in + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
			for (int x = 0; x < x_boundary; x += 32) {
				float *in_add_offset = in_add_offset_orig + x;
				float *padded_in_offset_3 = padded_in_offset_3_orig + x;
				_mm_storeu_ps(padded_in_offset_3, _mm_loadu_ps(in_add_offset));
				_mm_storeu_ps(padded_in_offset_3 + 4, _mm_loadu_ps(in_add_offset + 4));
				_mm_storeu_ps(padded_in_offset_3 + 8, _mm_loadu_ps(in_add_offset + 8));
				_mm_storeu_ps(padded_in_offset_3 + 12, _mm_loadu_ps(in_add_offset + 12));
				_mm_storeu_ps(padded_in_offset_3 + 16, _mm_loadu_ps(in_add_offset + 16));
				_mm_storeu_ps(padded_in_offset_3 + 20, _mm_loadu_ps(in_add_offset + 20));
				_mm_storeu_ps(padded_in_offset_3 + 24, _mm_loadu_ps(in_add_offset + 24));
				_mm_storeu_ps(padded_in_offset_3 + 28, _mm_loadu_ps(in_add_offset + 28));
			}
			for (int x = x_boundary; x < data_size_X; x++) {
				padded_in[(x + kern_cent_X) + (y + kern_cent_Y) * padded_data_size_X] = in[x + y * data_size_X];
			}
		}
    
    // main convolution loop
#pragma omp parallel for
    for(int y = 0; y < data_size_Y; y++) { // the y coordinate of theoutput location we're focusing on
      // because we know we have a minimum size of 400
      for(int x = 0; x < 1100; x += 100) { // the x coordinate of the output location we're focusing on
        __m128 sum_vec = _mm_setzero_ps();
        __m128 sum_vec2 = _mm_setzero_ps();
        __m128 sum_vec3 = _mm_setzero_ps();
        __m128 sum_vec4 = _mm_setzero_ps();
        __m128 sum_vec5 = _mm_setzero_ps();
        __m128 sum_vec6 = _mm_setzero_ps();
        __m128 sum_vec7 = _mm_setzero_ps();
        __m128 sum_vec8 = _mm_setzero_ps();
        __m128 sum_vec9 = _mm_setzero_ps();
        __m128 sum_vec10 = _mm_setzero_ps();
        __m128 sum_vec11 = _mm_setzero_ps();
        __m128 sum_vec12 = _mm_setzero_ps();
        __m128 sum_vec13 = _mm_setzero_ps();
        __m128 sum_vec14 = _mm_setzero_ps();
        __m128 sum_vec15 = _mm_setzero_ps();
        __m128 sum_vec16 = _mm_setzero_ps();
        __m128 sum_vec17 = _mm_setzero_ps();
        __m128 sum_vec18 = _mm_setzero_ps();
        __m128 sum_vec19 = _mm_setzero_ps();
        __m128 sum_vec20 = _mm_setzero_ps();
        __m128 sum_vec21 = _mm_setzero_ps();
        __m128 sum_vec22 = _mm_setzero_ps();
        __m128 sum_vec23 = _mm_setzero_ps();
        __m128 sum_vec24 = _mm_setzero_ps();
        __m128 sum_vec25 = _mm_setzero_ps();
        
        for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
          for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
            float *padded_in_Offset = padded_in + x + i + (y + j) *padded_data_size_X;
            __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
            
            __m128 input_vec = _mm_loadu_ps(padded_in_Offset);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
            __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset + 4);
            sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
            __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset + 8);
            sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
            __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset + 12);
            sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
            __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset + 16);
            sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
            __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset + 20);
            sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
            __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset + 24);
            sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
            __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset + 28);
            sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
            __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset + 32);
            sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
            __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset + 36);
            sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
            __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset + 40);
            sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
            __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset + 44);
            sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
            __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset + 48);
            sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
            __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset + 52);
            sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
            __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset + 56);
            sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
            __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset + 60);
            sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
            __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset + 64);
            sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
            __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset + 68);
            sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
            __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset + 72);
            sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
            __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset + 76);
            sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
            __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset + 80);
            sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
            __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset + 84);
            sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
            __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset + 88);
            sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
            __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset + 92);
            sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
            __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset + 96);
            sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
          }
        }
        float *outOffset = out + x + y * data_size_X;
        _mm_storeu_ps(outOffset, sum_vec);
        _mm_storeu_ps(outOffset + 4, sum_vec2);
        _mm_storeu_ps(outOffset + 8, sum_vec3);
        _mm_storeu_ps(outOffset + 12, sum_vec4);
        _mm_storeu_ps(outOffset + 16, sum_vec5);
        _mm_storeu_ps(outOffset + 20, sum_vec6);
        _mm_storeu_ps(outOffset + 24, sum_vec7);
        _mm_storeu_ps(outOffset + 28, sum_vec8);
        _mm_storeu_ps(outOffset + 32, sum_vec9);
        _mm_storeu_ps(outOffset + 36, sum_vec10);
        _mm_storeu_ps(outOffset + 40, sum_vec11);
        _mm_storeu_ps(outOffset + 44, sum_vec12);
        _mm_storeu_ps(outOffset + 48, sum_vec13);
        _mm_storeu_ps(outOffset + 52, sum_vec14);
        _mm_storeu_ps(outOffset + 56, sum_vec15);
        _mm_storeu_ps(outOffset + 60, sum_vec16);
        _mm_storeu_ps(outOffset + 64, sum_vec17);
        _mm_storeu_ps(outOffset + 68, sum_vec18);
        _mm_storeu_ps(outOffset + 72, sum_vec19);
        _mm_storeu_ps(outOffset + 76, sum_vec20);
        _mm_storeu_ps(outOffset + 80, sum_vec21);
        _mm_storeu_ps(outOffset + 84, sum_vec22);
        _mm_storeu_ps(outOffset + 88, sum_vec23);
        _mm_storeu_ps(outOffset + 92, sum_vec24);
        _mm_storeu_ps(outOffset + 96, sum_vec25);
			}
      
      // for sizes over 1100
      // for(int x = 1100; x < data_size_X; x += 100) { // the x coordinate of the output location we're focusing on
			int x = 1100;
      __m128 sum_vec = _mm_setzero_ps();
      __m128 sum_vec2 = _mm_setzero_ps();
      __m128 sum_vec3 = _mm_setzero_ps();
      __m128 sum_vec4 = _mm_setzero_ps();
      __m128 sum_vec5 = _mm_setzero_ps();
      __m128 sum_vec6 = _mm_setzero_ps();
      __m128 sum_vec7 = _mm_setzero_ps();
      __m128 sum_vec8 = _mm_setzero_ps();
      __m128 sum_vec9 = _mm_setzero_ps();
      __m128 sum_vec10 = _mm_setzero_ps();
      __m128 sum_vec11 = _mm_setzero_ps();
      __m128 sum_vec12 = _mm_setzero_ps();
      __m128 sum_vec13 = _mm_setzero_ps();
      __m128 sum_vec14 = _mm_setzero_ps();
      __m128 sum_vec15 = _mm_setzero_ps();
      __m128 sum_vec16 = _mm_setzero_ps();
      __m128 sum_vec17 = _mm_setzero_ps();
      __m128 sum_vec18 = _mm_setzero_ps();
      __m128 sum_vec19 = _mm_setzero_ps();
      __m128 sum_vec20 = _mm_setzero_ps();
      __m128 sum_vec21 = _mm_setzero_ps();
      __m128 sum_vec22 = _mm_setzero_ps();
      __m128 sum_vec23 = _mm_setzero_ps();
      __m128 sum_vec24 = _mm_setzero_ps();
      __m128 sum_vec25 = _mm_setzero_ps();
      
      for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
        for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
          
          float *padded_in_Offset2 = padded_in + x + i + (y + j) *padded_data_size_X;
          __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
          
          __m128 input_vec = _mm_loadu_ps(padded_in_Offset2);
          sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
          __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset2 + 4);
          sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
          __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset2 + 8);
          sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
          __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset2 + 12);
          sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
          __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset2 + 16);
          sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
          __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset2 + 20);
          sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
          __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset2 + 24);
          sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
          __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset2 + 28);
          sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
          __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset2 + 32);
          sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
          __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset2 + 36);
          sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
          __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset2 + 40);
          sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
          __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset2 + 44);
          sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
          __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset2 + 48);
          sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
          __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset2 + 52);
          sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
          __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset2 + 56);
          sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
          __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset2 + 60);
          sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
          __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset2 + 64);
          sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
          __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset2 + 68);
          sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
          __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset2 + 72);
          sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
          __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset2 + 76);
          sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
          __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset2 + 80);
          sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
          __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset2 + 84);
          sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
          __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset2 + 88);
          sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
          __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset2 + 92);
          sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
          __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset2 + 96);
          sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
        }
      }
      
      float *padded_out_ptr = padded_out + x + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      _mm_storeu_ps(padded_out_ptr, sum_vec);
      _mm_storeu_ps(padded_out_ptr + 4, sum_vec2);
      _mm_storeu_ps(padded_out_ptr + 8, sum_vec3);
      _mm_storeu_ps(padded_out_ptr + 12, sum_vec4);
      _mm_storeu_ps(padded_out_ptr + 16, sum_vec5);
      _mm_storeu_ps(padded_out_ptr + 20, sum_vec6);
      _mm_storeu_ps(padded_out_ptr + 24, sum_vec7);
      _mm_storeu_ps(padded_out_ptr + 28, sum_vec8);
      _mm_storeu_ps(padded_out_ptr + 32, sum_vec9);
      _mm_storeu_ps(padded_out_ptr + 36, sum_vec10);
      _mm_storeu_ps(padded_out_ptr + 40, sum_vec11);
      _mm_storeu_ps(padded_out_ptr + 44, sum_vec12);
      _mm_storeu_ps(padded_out_ptr + 48, sum_vec13);
      _mm_storeu_ps(padded_out_ptr + 52, sum_vec14);
      _mm_storeu_ps(padded_out_ptr + 56, sum_vec15);
      _mm_storeu_ps(padded_out_ptr + 60, sum_vec16);
      _mm_storeu_ps(padded_out_ptr + 64, sum_vec17);
      _mm_storeu_ps(padded_out_ptr + 68, sum_vec18);
      _mm_storeu_ps(padded_out_ptr + 72, sum_vec19);
      _mm_storeu_ps(padded_out_ptr + 76, sum_vec20);
      _mm_storeu_ps(padded_out_ptr + 80, sum_vec21);
      _mm_storeu_ps(padded_out_ptr + 84, sum_vec22);
      _mm_storeu_ps(padded_out_ptr + 88, sum_vec23);
      _mm_storeu_ps(padded_out_ptr + 92, sum_vec24);
      _mm_storeu_ps(padded_out_ptr + 96, sum_vec25);
    }
    
    int y_MUL_data_size_X;
    int y_PLUS_kern_MUL_pad;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      y_MUL_data_size_X = y * data_size_X;
      y_PLUS_kern_MUL_pad = (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 1100; x < data_size_X; x++) {
        out[x + y_MUL_data_size_X] =	padded_out[x + kern_cent_X + y_PLUS_kern_MUL_pad];
      }
    }
    
    /*****************************/
    /*    Greater than 1000      */
    /*****************************/
  } else if (data_size_X > 1000) {   
    static float padded_in[PADDEDMATRIXSIZE]__attribute__((aligned(16))); // 1204 * 1204  
    static int trials;
    trials++;
    
    if (trials > 1) {    
      // fill the preinput matrix with zeros
      int padded_size = (padded_data_size_X)* (padded_data_size_Y); // maybe remove or lower the padded_size bounds
      int padded_size_block = padded_size/100*100;
#pragma omp parallel for
      for (int z = 400; z < padded_size_block; z += 100) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 4 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 8 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 12 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 16 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 20 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 24 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 28 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 32 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 36 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 40 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 44 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 48 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 52 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 56 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 60 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 64 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 68 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 72 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 76 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 80 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 84 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 88 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 92 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 96 + z, _mm_setzero_ps());
      }
      for (int z = padded_size_block; z < padded_size; z += 4) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
      }
    }
   
    // Insert input matrix into padded matrix
    int x_boundary = data_size_X/32*32;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      float *in_add_offset_orig = in + y * data_size_X;
      float *padded_in_offset_3_orig = padded_in + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 0; x < x_boundary; x += 32) {
        float *in_add_offset = in_add_offset_orig + x;
        float *padded_in_offset_3 = padded_in_offset_3_orig + x;
        _mm_storeu_ps(padded_in_offset_3, _mm_loadu_ps(in_add_offset));
        _mm_storeu_ps(padded_in_offset_3 + 4, _mm_loadu_ps(in_add_offset + 4));
        _mm_storeu_ps(padded_in_offset_3 + 8, _mm_loadu_ps(in_add_offset + 8));
        _mm_storeu_ps(padded_in_offset_3 + 12, _mm_loadu_ps(in_add_offset + 12));
        _mm_storeu_ps(padded_in_offset_3 + 16, _mm_loadu_ps(in_add_offset + 16));
        _mm_storeu_ps(padded_in_offset_3 + 20, _mm_loadu_ps(in_add_offset + 20));
        _mm_storeu_ps(padded_in_offset_3 + 24, _mm_loadu_ps(in_add_offset + 24));
        _mm_storeu_ps(padded_in_offset_3 + 28, _mm_loadu_ps(in_add_offset + 28));
      }
      for (int x = x_boundary; x < data_size_X; x++) {
        padded_in[(x + kern_cent_X) + (y + kern_cent_Y) * padded_data_size_X] = in[x + y * data_size_X];
      }
    }
    
    // main convolution loop
#pragma omp parallel for
    for(int y = 0; y < data_size_Y; y++) { // the y coordinate of theoutput location we're focusing on
      // because we know we have a minimum size of 400
      for(int x = 0; x < 1000; x += 100) { // the x coordinate of the output location we're focusing on
        __m128 sum_vec = _mm_setzero_ps();
        __m128 sum_vec2 = _mm_setzero_ps();
        __m128 sum_vec3 = _mm_setzero_ps();
        __m128 sum_vec4 = _mm_setzero_ps();
        __m128 sum_vec5 = _mm_setzero_ps();
        __m128 sum_vec6 = _mm_setzero_ps();
        __m128 sum_vec7 = _mm_setzero_ps();
        __m128 sum_vec8 = _mm_setzero_ps();
        __m128 sum_vec9 = _mm_setzero_ps();
        __m128 sum_vec10 = _mm_setzero_ps();
        __m128 sum_vec11 = _mm_setzero_ps();
        __m128 sum_vec12 = _mm_setzero_ps();
        __m128 sum_vec13 = _mm_setzero_ps();
        __m128 sum_vec14 = _mm_setzero_ps();
        __m128 sum_vec15 = _mm_setzero_ps();
        __m128 sum_vec16 = _mm_setzero_ps();
        __m128 sum_vec17 = _mm_setzero_ps();
        __m128 sum_vec18 = _mm_setzero_ps();
        __m128 sum_vec19 = _mm_setzero_ps();
        __m128 sum_vec20 = _mm_setzero_ps();
        __m128 sum_vec21 = _mm_setzero_ps();
        __m128 sum_vec22 = _mm_setzero_ps();
        __m128 sum_vec23 = _mm_setzero_ps();
        __m128 sum_vec24 = _mm_setzero_ps();
        __m128 sum_vec25 = _mm_setzero_ps();
        
        for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
          for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
            float *padded_in_Offset = padded_in + x + i + (y + j) *padded_data_size_X;
            __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
            
            __m128 input_vec = _mm_loadu_ps(padded_in_Offset);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
            __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset + 4);
            sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
            __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset + 8);
            sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
            __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset + 12);
            sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
            __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset + 16);
            sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
            __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset + 20);
            sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
            __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset + 24);
            sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
            __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset + 28);
            sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
            __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset + 32);
            sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
            __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset + 36);
            sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
            __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset + 40);
            sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
            __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset + 44);
            sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
            __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset + 48);
            sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
            __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset + 52);
            sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
            __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset + 56);
            sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
            __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset + 60);
            sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
            __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset + 64);
            sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
            __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset + 68);
            sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
            __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset + 72);
            sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
            __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset + 76);
            sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
            __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset + 80);
            sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
            __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset + 84);
            sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
            __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset + 88);
            sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
            __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset + 92);
            sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
            __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset + 96);
            sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
          }
        }
        float *outOffset = out + x + y * data_size_X;
        _mm_storeu_ps(outOffset, sum_vec);
        _mm_storeu_ps(outOffset + 4, sum_vec2);
        _mm_storeu_ps(outOffset + 8, sum_vec3);
        _mm_storeu_ps(outOffset + 12, sum_vec4);
        _mm_storeu_ps(outOffset + 16, sum_vec5);
        _mm_storeu_ps(outOffset + 20, sum_vec6);
        _mm_storeu_ps(outOffset + 24, sum_vec7);
        _mm_storeu_ps(outOffset + 28, sum_vec8);
        _mm_storeu_ps(outOffset + 32, sum_vec9);
        _mm_storeu_ps(outOffset + 36, sum_vec10);
        _mm_storeu_ps(outOffset + 40, sum_vec11);
        _mm_storeu_ps(outOffset + 44, sum_vec12);
        _mm_storeu_ps(outOffset + 48, sum_vec13);
        _mm_storeu_ps(outOffset + 52, sum_vec14);
        _mm_storeu_ps(outOffset + 56, sum_vec15);
        _mm_storeu_ps(outOffset + 60, sum_vec16);
        _mm_storeu_ps(outOffset + 64, sum_vec17);
        _mm_storeu_ps(outOffset + 68, sum_vec18);
        _mm_storeu_ps(outOffset + 72, sum_vec19);
        _mm_storeu_ps(outOffset + 76, sum_vec20);
        _mm_storeu_ps(outOffset + 80, sum_vec21);
        _mm_storeu_ps(outOffset + 84, sum_vec22);
        _mm_storeu_ps(outOffset + 88, sum_vec23);
        _mm_storeu_ps(outOffset + 92, sum_vec24);
        _mm_storeu_ps(outOffset + 96, sum_vec25);
			}
      
      // for sizes over 1000
			int x = 1000;
      __m128 sum_vec = _mm_setzero_ps();
      __m128 sum_vec2 = _mm_setzero_ps();
      __m128 sum_vec3 = _mm_setzero_ps();
      __m128 sum_vec4 = _mm_setzero_ps();
      __m128 sum_vec5 = _mm_setzero_ps();
      __m128 sum_vec6 = _mm_setzero_ps();
      __m128 sum_vec7 = _mm_setzero_ps();
      __m128 sum_vec8 = _mm_setzero_ps();
      __m128 sum_vec9 = _mm_setzero_ps();
      __m128 sum_vec10 = _mm_setzero_ps();
      __m128 sum_vec11 = _mm_setzero_ps();
      __m128 sum_vec12 = _mm_setzero_ps();
      __m128 sum_vec13 = _mm_setzero_ps();
      __m128 sum_vec14 = _mm_setzero_ps();
      __m128 sum_vec15 = _mm_setzero_ps();
      __m128 sum_vec16 = _mm_setzero_ps();
      __m128 sum_vec17 = _mm_setzero_ps();
      __m128 sum_vec18 = _mm_setzero_ps();
      __m128 sum_vec19 = _mm_setzero_ps();
      __m128 sum_vec20 = _mm_setzero_ps();
      __m128 sum_vec21 = _mm_setzero_ps();
      __m128 sum_vec22 = _mm_setzero_ps();
      __m128 sum_vec23 = _mm_setzero_ps();
      __m128 sum_vec24 = _mm_setzero_ps();
      __m128 sum_vec25 = _mm_setzero_ps();
      
      for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
        for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
          
          float *padded_in_Offset2 = padded_in + x + i + (y + j) *padded_data_size_X;
          __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
          
          __m128 input_vec = _mm_loadu_ps(padded_in_Offset2);
          sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
          __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset2 + 4);
          sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
          __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset2 + 8);
          sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
          __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset2 + 12);
          sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
          __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset2 + 16);
          sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
          __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset2 + 20);
          sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
          __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset2 + 24);
          sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
          __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset2 + 28);
          sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
          __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset2 + 32);
          sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
          __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset2 + 36);
          sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
          __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset2 + 40);
          sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
          __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset2 + 44);
          sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
          __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset2 + 48);
          sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
          __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset2 + 52);
          sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
          __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset2 + 56);
          sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
          __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset2 + 60);
          sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
          __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset2 + 64);
          sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
          __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset2 + 68);
          sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
          __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset2 + 72);
          sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
          __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset2 + 76);
          sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
          __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset2 + 80);
          sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
          __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset2 + 84);
          sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
          __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset2 + 88);
          sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
          __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset2 + 92);
          sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
          __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset2 + 96);
          sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
        }
      }
      
      float *padded_out_ptr = padded_out + x + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      _mm_storeu_ps(padded_out_ptr, sum_vec);
      _mm_storeu_ps(padded_out_ptr + 4, sum_vec2);
      _mm_storeu_ps(padded_out_ptr + 8, sum_vec3);
      _mm_storeu_ps(padded_out_ptr + 12, sum_vec4);
      _mm_storeu_ps(padded_out_ptr + 16, sum_vec5);
      _mm_storeu_ps(padded_out_ptr + 20, sum_vec6);
      _mm_storeu_ps(padded_out_ptr + 24, sum_vec7);
      _mm_storeu_ps(padded_out_ptr + 28, sum_vec8);
      _mm_storeu_ps(padded_out_ptr + 32, sum_vec9);
      _mm_storeu_ps(padded_out_ptr + 36, sum_vec10);
      _mm_storeu_ps(padded_out_ptr + 40, sum_vec11);
      _mm_storeu_ps(padded_out_ptr + 44, sum_vec12);
      _mm_storeu_ps(padded_out_ptr + 48, sum_vec13);
      _mm_storeu_ps(padded_out_ptr + 52, sum_vec14);
      _mm_storeu_ps(padded_out_ptr + 56, sum_vec15);
      _mm_storeu_ps(padded_out_ptr + 60, sum_vec16);
      _mm_storeu_ps(padded_out_ptr + 64, sum_vec17);
      _mm_storeu_ps(padded_out_ptr + 68, sum_vec18);
      _mm_storeu_ps(padded_out_ptr + 72, sum_vec19);
      _mm_storeu_ps(padded_out_ptr + 76, sum_vec20);
      _mm_storeu_ps(padded_out_ptr + 80, sum_vec21);
      _mm_storeu_ps(padded_out_ptr + 84, sum_vec22);
      _mm_storeu_ps(padded_out_ptr + 88, sum_vec23);
      _mm_storeu_ps(padded_out_ptr + 92, sum_vec24);
      _mm_storeu_ps(padded_out_ptr + 96, sum_vec25);
    }
    
    int y_MUL_data_size_X;
    int y_PLUS_kern_MUL_pad;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      y_MUL_data_size_X = y * data_size_X;
      y_PLUS_kern_MUL_pad = (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 1000; x < data_size_X; x++) {
        out[x + y_MUL_data_size_X] =	padded_out[x + kern_cent_X + y_PLUS_kern_MUL_pad];
      }
    }
    
    /*****************************/
    /*    Greater than 900       */
    /*****************************/
  } else if (data_size_X > 900) {	
    static float padded_in[PADDEDMATRIXSIZE]__attribute__((aligned(16))); // 1204 * 1204   
    static int trials;
    trials++;
    
    if (trials > 1) {   
      // fill the preinput matrix with zeros
      int padded_size = (padded_data_size_X)* (padded_data_size_Y); // maybe remove or lower the padded_size bounds
      int padded_size_block = padded_size/100*100;
#pragma omp parallel for
      for (int z = 400; z < padded_size_block; z += 100) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 4 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 8 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 12 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 16 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 20 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 24 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 28 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 32 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 36 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 40 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 44 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 48 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 52 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 56 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 60 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 64 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 68 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 72 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 76 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 80 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 84 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 88 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 92 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 96 + z, _mm_setzero_ps());
      }
      for (int z = padded_size_block; z < padded_size; z += 4) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
      }
    }

    // Insert input matrix into padded matrix
    int x_boundary = data_size_X/32*32;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      float *in_add_offset_orig = in + y * data_size_X;
      float *padded_in_offset_3_orig = padded_in + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 0; x < x_boundary; x += 32) {
        float *in_add_offset = in_add_offset_orig + x;
        float *padded_in_offset_3 = padded_in_offset_3_orig + x;
        _mm_storeu_ps(padded_in_offset_3, _mm_loadu_ps(in_add_offset));
        _mm_storeu_ps(padded_in_offset_3 + 4, _mm_loadu_ps(in_add_offset + 4));
        _mm_storeu_ps(padded_in_offset_3 + 8, _mm_loadu_ps(in_add_offset + 8));
        _mm_storeu_ps(padded_in_offset_3 + 12, _mm_loadu_ps(in_add_offset + 12));
        _mm_storeu_ps(padded_in_offset_3 + 16, _mm_loadu_ps(in_add_offset + 16));
        _mm_storeu_ps(padded_in_offset_3 + 20, _mm_loadu_ps(in_add_offset + 20));
        _mm_storeu_ps(padded_in_offset_3 + 24, _mm_loadu_ps(in_add_offset + 24));
        _mm_storeu_ps(padded_in_offset_3 + 28, _mm_loadu_ps(in_add_offset + 28));
      }
      for (int x = x_boundary; x < data_size_X; x++) {
        padded_in[(x + kern_cent_X) + (y + kern_cent_Y) * padded_data_size_X] = in[x + y * data_size_X];
      }
    }
    
    // main convolution loop
#pragma omp parallel for
    for(int y = 0; y < data_size_Y; y++) { // the y coordinate of theoutput location we're focusing on
      // because we know we have a minimum size of 400
      for(int x = 0; x < 900; x += 100) { // the x coordinate of the output location we're focusing on
        __m128 sum_vec = _mm_setzero_ps();
        __m128 sum_vec2 = _mm_setzero_ps();
        __m128 sum_vec3 = _mm_setzero_ps();
        __m128 sum_vec4 = _mm_setzero_ps();
        __m128 sum_vec5 = _mm_setzero_ps();
        __m128 sum_vec6 = _mm_setzero_ps();
        __m128 sum_vec7 = _mm_setzero_ps();
        __m128 sum_vec8 = _mm_setzero_ps();
        __m128 sum_vec9 = _mm_setzero_ps();
        __m128 sum_vec10 = _mm_setzero_ps();
        __m128 sum_vec11 = _mm_setzero_ps();
        __m128 sum_vec12 = _mm_setzero_ps();
        __m128 sum_vec13 = _mm_setzero_ps();
        __m128 sum_vec14 = _mm_setzero_ps();
        __m128 sum_vec15 = _mm_setzero_ps();
        __m128 sum_vec16 = _mm_setzero_ps();
        __m128 sum_vec17 = _mm_setzero_ps();
        __m128 sum_vec18 = _mm_setzero_ps();
        __m128 sum_vec19 = _mm_setzero_ps();
        __m128 sum_vec20 = _mm_setzero_ps();
        __m128 sum_vec21 = _mm_setzero_ps();
        __m128 sum_vec22 = _mm_setzero_ps();
        __m128 sum_vec23 = _mm_setzero_ps();
        __m128 sum_vec24 = _mm_setzero_ps();
        __m128 sum_vec25 = _mm_setzero_ps();
        
        for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
          for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
            float *padded_in_Offset = padded_in + x + i + (y + j) *padded_data_size_X;
            __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
            
            __m128 input_vec = _mm_loadu_ps(padded_in_Offset);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
            __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset + 4);
            sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
            __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset + 8);
            sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
            __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset + 12);
            sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
            __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset + 16);
            sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
            __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset + 20);
            sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
            __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset + 24);
            sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
            __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset + 28);
            sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
            __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset + 32);
            sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
            __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset + 36);
            sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
            __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset + 40);
            sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
            __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset + 44);
            sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
            __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset + 48);
            sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
            __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset + 52);
            sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
            __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset + 56);
            sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
            __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset + 60);
            sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
            __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset + 64);
            sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
            __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset + 68);
            sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
            __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset + 72);
            sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
            __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset + 76);
            sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
            __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset + 80);
            sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
            __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset + 84);
            sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
            __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset + 88);
            sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
            __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset + 92);
            sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
            __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset + 96);
            sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
          }
        }
        float *outOffset = out + x + y * data_size_X;
        _mm_storeu_ps(outOffset, sum_vec);
        _mm_storeu_ps(outOffset + 4, sum_vec2);
        _mm_storeu_ps(outOffset + 8, sum_vec3);
        _mm_storeu_ps(outOffset + 12, sum_vec4);
        _mm_storeu_ps(outOffset + 16, sum_vec5);
        _mm_storeu_ps(outOffset + 20, sum_vec6);
        _mm_storeu_ps(outOffset + 24, sum_vec7);
        _mm_storeu_ps(outOffset + 28, sum_vec8);
        _mm_storeu_ps(outOffset + 32, sum_vec9);
        _mm_storeu_ps(outOffset + 36, sum_vec10);
        _mm_storeu_ps(outOffset + 40, sum_vec11);
        _mm_storeu_ps(outOffset + 44, sum_vec12);
        _mm_storeu_ps(outOffset + 48, sum_vec13);
        _mm_storeu_ps(outOffset + 52, sum_vec14);
        _mm_storeu_ps(outOffset + 56, sum_vec15);
        _mm_storeu_ps(outOffset + 60, sum_vec16);
        _mm_storeu_ps(outOffset + 64, sum_vec17);
        _mm_storeu_ps(outOffset + 68, sum_vec18);
        _mm_storeu_ps(outOffset + 72, sum_vec19);
        _mm_storeu_ps(outOffset + 76, sum_vec20);
        _mm_storeu_ps(outOffset + 80, sum_vec21);
        _mm_storeu_ps(outOffset + 84, sum_vec22);
        _mm_storeu_ps(outOffset + 88, sum_vec23);
        _mm_storeu_ps(outOffset + 92, sum_vec24);
        _mm_storeu_ps(outOffset + 96, sum_vec25);
			}
      
      // for sizes over 900
			int x = 900;
      __m128 sum_vec = _mm_setzero_ps();
      __m128 sum_vec2 = _mm_setzero_ps();
      __m128 sum_vec3 = _mm_setzero_ps();
      __m128 sum_vec4 = _mm_setzero_ps();
      __m128 sum_vec5 = _mm_setzero_ps();
      __m128 sum_vec6 = _mm_setzero_ps();
      __m128 sum_vec7 = _mm_setzero_ps();
      __m128 sum_vec8 = _mm_setzero_ps();
      __m128 sum_vec9 = _mm_setzero_ps();
      __m128 sum_vec10 = _mm_setzero_ps();
      __m128 sum_vec11 = _mm_setzero_ps();
      __m128 sum_vec12 = _mm_setzero_ps();
      __m128 sum_vec13 = _mm_setzero_ps();
      __m128 sum_vec14 = _mm_setzero_ps();
      __m128 sum_vec15 = _mm_setzero_ps();
      __m128 sum_vec16 = _mm_setzero_ps();
      __m128 sum_vec17 = _mm_setzero_ps();
      __m128 sum_vec18 = _mm_setzero_ps();
      __m128 sum_vec19 = _mm_setzero_ps();
      __m128 sum_vec20 = _mm_setzero_ps();
      __m128 sum_vec21 = _mm_setzero_ps();
      __m128 sum_vec22 = _mm_setzero_ps();
      __m128 sum_vec23 = _mm_setzero_ps();
      __m128 sum_vec24 = _mm_setzero_ps();
      __m128 sum_vec25 = _mm_setzero_ps();
      
      for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
        for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
          
          float *padded_in_Offset2 = padded_in + x + i + (y + j) *padded_data_size_X;
          __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
          
          __m128 input_vec = _mm_loadu_ps(padded_in_Offset2);
          sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
          __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset2 + 4);
          sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
          __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset2 + 8);
          sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
          __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset2 + 12);
          sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
          __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset2 + 16);
          sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
          __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset2 + 20);
          sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
          __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset2 + 24);
          sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
          __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset2 + 28);
          sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
          __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset2 + 32);
          sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
          __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset2 + 36);
          sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
          __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset2 + 40);
          sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
          __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset2 + 44);
          sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
          __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset2 + 48);
          sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
          __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset2 + 52);
          sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
          __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset2 + 56);
          sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
          __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset2 + 60);
          sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
          __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset2 + 64);
          sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
          __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset2 + 68);
          sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
          __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset2 + 72);
          sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
          __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset2 + 76);
          sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
          __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset2 + 80);
          sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
          __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset2 + 84);
          sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
          __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset2 + 88);
          sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
          __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset2 + 92);
          sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
          __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset2 + 96);
          sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
        }
      }
      
      float *padded_out_ptr = padded_out + x + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      _mm_storeu_ps(padded_out_ptr, sum_vec);
      _mm_storeu_ps(padded_out_ptr + 4, sum_vec2);
      _mm_storeu_ps(padded_out_ptr + 8, sum_vec3);
      _mm_storeu_ps(padded_out_ptr + 12, sum_vec4);
      _mm_storeu_ps(padded_out_ptr + 16, sum_vec5);
      _mm_storeu_ps(padded_out_ptr + 20, sum_vec6);
      _mm_storeu_ps(padded_out_ptr + 24, sum_vec7);
      _mm_storeu_ps(padded_out_ptr + 28, sum_vec8);
      _mm_storeu_ps(padded_out_ptr + 32, sum_vec9);
      _mm_storeu_ps(padded_out_ptr + 36, sum_vec10);
      _mm_storeu_ps(padded_out_ptr + 40, sum_vec11);
      _mm_storeu_ps(padded_out_ptr + 44, sum_vec12);
      _mm_storeu_ps(padded_out_ptr + 48, sum_vec13);
      _mm_storeu_ps(padded_out_ptr + 52, sum_vec14);
      _mm_storeu_ps(padded_out_ptr + 56, sum_vec15);
      _mm_storeu_ps(padded_out_ptr + 60, sum_vec16);
      _mm_storeu_ps(padded_out_ptr + 64, sum_vec17);
      _mm_storeu_ps(padded_out_ptr + 68, sum_vec18);
      _mm_storeu_ps(padded_out_ptr + 72, sum_vec19);
      _mm_storeu_ps(padded_out_ptr + 76, sum_vec20);
      _mm_storeu_ps(padded_out_ptr + 80, sum_vec21);
      _mm_storeu_ps(padded_out_ptr + 84, sum_vec22);
      _mm_storeu_ps(padded_out_ptr + 88, sum_vec23);
      _mm_storeu_ps(padded_out_ptr + 92, sum_vec24);
      _mm_storeu_ps(padded_out_ptr + 96, sum_vec25);
    }
    
    int y_MUL_data_size_X;
    int y_PLUS_kern_MUL_pad;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      y_MUL_data_size_X = y * data_size_X;
      y_PLUS_kern_MUL_pad = (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 900; x < data_size_X; x++) {
        out[x + y_MUL_data_size_X] =	padded_out[x + kern_cent_X + y_PLUS_kern_MUL_pad];
      }
    }
  
    /*****************************/
    /*    Greater than 800       */
    /*****************************/
  } else if (data_size_X > 800) {
    static float padded_in[PADDEDMATRIXSIZE]__attribute__((aligned(16))); // 1204 * 1204
    static int trials;
    trials++;
    
    if (trials > 1) { 
      // fill the preinput matrix with zeros
      int padded_size = (padded_data_size_X)* (padded_data_size_Y); // maybe remove or lower the padded_size bounds
      int padded_size_block = padded_size/100*100;
#pragma omp parallel for
      for (int z = 400; z < padded_size_block; z += 100) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 4 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 8 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 12 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 16 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 20 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 24 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 28 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 32 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 36 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 40 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 44 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 48 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 52 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 56 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 60 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 64 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 68 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 72 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 76 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 80 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 84 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 88 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 92 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 96 + z, _mm_setzero_ps());
      }
      for (int z = padded_size_block; z < padded_size; z += 4) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
      }
    }

    // Insert input matrix into padded matrix
    int x_boundary = data_size_X/32*32;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      float *in_add_offset_orig = in + y * data_size_X;
      float *padded_in_offset_3_orig = padded_in + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 0; x < x_boundary; x += 32) {
        float *in_add_offset = in_add_offset_orig + x;
        float *padded_in_offset_3 = padded_in_offset_3_orig + x;
        _mm_storeu_ps(padded_in_offset_3, _mm_loadu_ps(in_add_offset));
        _mm_storeu_ps(padded_in_offset_3 + 4, _mm_loadu_ps(in_add_offset + 4));
        _mm_storeu_ps(padded_in_offset_3 + 8, _mm_loadu_ps(in_add_offset + 8));
        _mm_storeu_ps(padded_in_offset_3 + 12, _mm_loadu_ps(in_add_offset + 12));
        _mm_storeu_ps(padded_in_offset_3 + 16, _mm_loadu_ps(in_add_offset + 16));
        _mm_storeu_ps(padded_in_offset_3 + 20, _mm_loadu_ps(in_add_offset + 20));
        _mm_storeu_ps(padded_in_offset_3 + 24, _mm_loadu_ps(in_add_offset + 24));
        _mm_storeu_ps(padded_in_offset_3 + 28, _mm_loadu_ps(in_add_offset + 28));
      }
      for (int x = x_boundary; x < data_size_X; x++) {
        padded_in[(x + kern_cent_X) + (y + kern_cent_Y) * padded_data_size_X] = in[x + y * data_size_X];
      }
    }
    
    // main convolution loop
#pragma omp parallel for
    for(int y = 0; y < data_size_Y; y++) { // the y coordinate of theoutput location we're focusing on
      for(int x = 0; x < 800; x += 100) { // the x coordinate of the output location we're focusing on
        __m128 sum_vec = _mm_setzero_ps();
        __m128 sum_vec2 = _mm_setzero_ps();
        __m128 sum_vec3 = _mm_setzero_ps();
        __m128 sum_vec4 = _mm_setzero_ps();
        __m128 sum_vec5 = _mm_setzero_ps();
        __m128 sum_vec6 = _mm_setzero_ps();
        __m128 sum_vec7 = _mm_setzero_ps();
        __m128 sum_vec8 = _mm_setzero_ps();
        __m128 sum_vec9 = _mm_setzero_ps();
        __m128 sum_vec10 = _mm_setzero_ps();
        __m128 sum_vec11 = _mm_setzero_ps();
        __m128 sum_vec12 = _mm_setzero_ps();
        __m128 sum_vec13 = _mm_setzero_ps();
        __m128 sum_vec14 = _mm_setzero_ps();
        __m128 sum_vec15 = _mm_setzero_ps();
        __m128 sum_vec16 = _mm_setzero_ps();
        __m128 sum_vec17 = _mm_setzero_ps();
        __m128 sum_vec18 = _mm_setzero_ps();
        __m128 sum_vec19 = _mm_setzero_ps();
        __m128 sum_vec20 = _mm_setzero_ps();
        __m128 sum_vec21 = _mm_setzero_ps();
        __m128 sum_vec22 = _mm_setzero_ps();
        __m128 sum_vec23 = _mm_setzero_ps();
        __m128 sum_vec24 = _mm_setzero_ps();
        __m128 sum_vec25 = _mm_setzero_ps();
        
        for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
          for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
            float *padded_in_Offset = padded_in + x + i + (y + j) *padded_data_size_X;
            __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
            
            __m128 input_vec = _mm_loadu_ps(padded_in_Offset);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
            __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset + 4);
            sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
            __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset + 8);
            sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
            __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset + 12);
            sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
            __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset + 16);
            sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
            __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset + 20);
            sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
            __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset + 24);
            sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
            __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset + 28);
            sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
            __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset + 32);
            sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
            __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset + 36);
            sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
            __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset + 40);
            sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
            __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset + 44);
            sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
            __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset + 48);
            sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
            __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset + 52);
            sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
            __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset + 56);
            sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
            __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset + 60);
            sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
            __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset + 64);
            sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
            __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset + 68);
            sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
            __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset + 72);
            sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
            __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset + 76);
            sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
            __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset + 80);
            sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
            __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset + 84);
            sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
            __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset + 88);
            sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
            __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset + 92);
            sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
            __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset + 96);
            sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
          }
        }
        float *outOffset = out + x + y * data_size_X;
        _mm_storeu_ps(outOffset, sum_vec);
        _mm_storeu_ps(outOffset + 4, sum_vec2);
        _mm_storeu_ps(outOffset + 8, sum_vec3);
        _mm_storeu_ps(outOffset + 12, sum_vec4);
        _mm_storeu_ps(outOffset + 16, sum_vec5);
        _mm_storeu_ps(outOffset + 20, sum_vec6);
        _mm_storeu_ps(outOffset + 24, sum_vec7);
        _mm_storeu_ps(outOffset + 28, sum_vec8);
        _mm_storeu_ps(outOffset + 32, sum_vec9);
        _mm_storeu_ps(outOffset + 36, sum_vec10);
        _mm_storeu_ps(outOffset + 40, sum_vec11);
        _mm_storeu_ps(outOffset + 44, sum_vec12);
        _mm_storeu_ps(outOffset + 48, sum_vec13);
        _mm_storeu_ps(outOffset + 52, sum_vec14);
        _mm_storeu_ps(outOffset + 56, sum_vec15);
        _mm_storeu_ps(outOffset + 60, sum_vec16);
        _mm_storeu_ps(outOffset + 64, sum_vec17);
        _mm_storeu_ps(outOffset + 68, sum_vec18);
        _mm_storeu_ps(outOffset + 72, sum_vec19);
        _mm_storeu_ps(outOffset + 76, sum_vec20);
        _mm_storeu_ps(outOffset + 80, sum_vec21);
        _mm_storeu_ps(outOffset + 84, sum_vec22);
        _mm_storeu_ps(outOffset + 88, sum_vec23);
        _mm_storeu_ps(outOffset + 92, sum_vec24);
        _mm_storeu_ps(outOffset + 96, sum_vec25);
			}
      
      // for sizes over 800
			int x = 800;
      __m128 sum_vec = _mm_setzero_ps();
      __m128 sum_vec2 = _mm_setzero_ps();
      __m128 sum_vec3 = _mm_setzero_ps();
      __m128 sum_vec4 = _mm_setzero_ps();
      __m128 sum_vec5 = _mm_setzero_ps();
      __m128 sum_vec6 = _mm_setzero_ps();
      __m128 sum_vec7 = _mm_setzero_ps();
      __m128 sum_vec8 = _mm_setzero_ps();
      __m128 sum_vec9 = _mm_setzero_ps();
      __m128 sum_vec10 = _mm_setzero_ps();
      __m128 sum_vec11 = _mm_setzero_ps();
      __m128 sum_vec12 = _mm_setzero_ps();
      __m128 sum_vec13 = _mm_setzero_ps();
      __m128 sum_vec14 = _mm_setzero_ps();
      __m128 sum_vec15 = _mm_setzero_ps();
      __m128 sum_vec16 = _mm_setzero_ps();
      __m128 sum_vec17 = _mm_setzero_ps();
      __m128 sum_vec18 = _mm_setzero_ps();
      __m128 sum_vec19 = _mm_setzero_ps();
      __m128 sum_vec20 = _mm_setzero_ps();
      __m128 sum_vec21 = _mm_setzero_ps();
      __m128 sum_vec22 = _mm_setzero_ps();
      __m128 sum_vec23 = _mm_setzero_ps();
      __m128 sum_vec24 = _mm_setzero_ps();
      __m128 sum_vec25 = _mm_setzero_ps();
      
      for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
        for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
          
          float *padded_in_Offset2 = padded_in + x + i + (y + j) *padded_data_size_X;
          __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
          
          __m128 input_vec = _mm_loadu_ps(padded_in_Offset2);
          sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
          __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset2 + 4);
          sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
          __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset2 + 8);
          sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
          __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset2 + 12);
          sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
          __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset2 + 16);
          sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
          __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset2 + 20);
          sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
          __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset2 + 24);
          sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
          __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset2 + 28);
          sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
          __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset2 + 32);
          sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
          __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset2 + 36);
          sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
          __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset2 + 40);
          sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
          __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset2 + 44);
          sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
          __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset2 + 48);
          sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
          __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset2 + 52);
          sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
          __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset2 + 56);
          sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
          __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset2 + 60);
          sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
          __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset2 + 64);
          sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
          __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset2 + 68);
          sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
          __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset2 + 72);
          sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
          __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset2 + 76);
          sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
          __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset2 + 80);
          sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
          __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset2 + 84);
          sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
          __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset2 + 88);
          sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
          __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset2 + 92);
          sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
          __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset2 + 96);
          sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
        }
      }
      
      float *padded_out_ptr = padded_out + x + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      _mm_storeu_ps(padded_out_ptr, sum_vec);
      _mm_storeu_ps(padded_out_ptr + 4, sum_vec2);
      _mm_storeu_ps(padded_out_ptr + 8, sum_vec3);
      _mm_storeu_ps(padded_out_ptr + 12, sum_vec4);
      _mm_storeu_ps(padded_out_ptr + 16, sum_vec5);
      _mm_storeu_ps(padded_out_ptr + 20, sum_vec6);
      _mm_storeu_ps(padded_out_ptr + 24, sum_vec7);
      _mm_storeu_ps(padded_out_ptr + 28, sum_vec8);
      _mm_storeu_ps(padded_out_ptr + 32, sum_vec9);
      _mm_storeu_ps(padded_out_ptr + 36, sum_vec10);
      _mm_storeu_ps(padded_out_ptr + 40, sum_vec11);
      _mm_storeu_ps(padded_out_ptr + 44, sum_vec12);
      _mm_storeu_ps(padded_out_ptr + 48, sum_vec13);
      _mm_storeu_ps(padded_out_ptr + 52, sum_vec14);
      _mm_storeu_ps(padded_out_ptr + 56, sum_vec15);
      _mm_storeu_ps(padded_out_ptr + 60, sum_vec16);
      _mm_storeu_ps(padded_out_ptr + 64, sum_vec17);
      _mm_storeu_ps(padded_out_ptr + 68, sum_vec18);
      _mm_storeu_ps(padded_out_ptr + 72, sum_vec19);
      _mm_storeu_ps(padded_out_ptr + 76, sum_vec20);
      _mm_storeu_ps(padded_out_ptr + 80, sum_vec21);
      _mm_storeu_ps(padded_out_ptr + 84, sum_vec22);
      _mm_storeu_ps(padded_out_ptr + 88, sum_vec23);
      _mm_storeu_ps(padded_out_ptr + 92, sum_vec24);
      _mm_storeu_ps(padded_out_ptr + 96, sum_vec25);
    }
    
    int y_MUL_data_size_X;
    int y_PLUS_kern_MUL_pad;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      y_MUL_data_size_X = y * data_size_X;
      y_PLUS_kern_MUL_pad = (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 800; x < data_size_X; x++) {
        out[x + y_MUL_data_size_X] =	padded_out[x + kern_cent_X + y_PLUS_kern_MUL_pad];
      }
    }
	 
    /*****************************/
    /*    Greater than 700       */
    /*****************************/
  } else if (data_size_X > 700) {
    static float padded_in[PADDEDMATRIXSIZE]__attribute__((aligned(16))); // 1204 * 1204   
    static int trials;
    trials++;
    
    if (trials > 1) { 
      // fill the preinput matrix with zeros
      int padded_size = (padded_data_size_X)* (padded_data_size_Y); // maybe remove or lower the padded_size bounds
      int padded_size_block = padded_size/100*100;
#pragma omp parallel for
      for (int z = 400; z < padded_size_block; z += 100) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 4 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 8 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 12 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 16 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 20 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 24 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 28 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 32 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 36 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 40 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 44 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 48 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 52 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 56 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 60 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 64 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 68 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 72 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 76 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 80 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 84 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 88 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 92 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 96 + z, _mm_setzero_ps());
      }
      for (int z = padded_size_block; z < padded_size; z += 4) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
      }
    }

    // Insert input matrix into padded matrix
    int x_boundary = data_size_X/32*32;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      float *in_add_offset_orig = in + y * data_size_X;
      float *padded_in_offset_3_orig = padded_in + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 0; x < x_boundary; x += 32) {
        float *in_add_offset = in_add_offset_orig + x;
        float *padded_in_offset_3 = padded_in_offset_3_orig + x;
        _mm_storeu_ps(padded_in_offset_3, _mm_loadu_ps(in_add_offset));
        _mm_storeu_ps(padded_in_offset_3 + 4, _mm_loadu_ps(in_add_offset + 4));
        _mm_storeu_ps(padded_in_offset_3 + 8, _mm_loadu_ps(in_add_offset + 8));
        _mm_storeu_ps(padded_in_offset_3 + 12, _mm_loadu_ps(in_add_offset + 12));
        _mm_storeu_ps(padded_in_offset_3 + 16, _mm_loadu_ps(in_add_offset + 16));
        _mm_storeu_ps(padded_in_offset_3 + 20, _mm_loadu_ps(in_add_offset + 20));
        _mm_storeu_ps(padded_in_offset_3 + 24, _mm_loadu_ps(in_add_offset + 24));
        _mm_storeu_ps(padded_in_offset_3 + 28, _mm_loadu_ps(in_add_offset + 28));
      }
      for (int x = x_boundary; x < data_size_X; x++) {
        padded_in[(x + kern_cent_X) + (y + kern_cent_Y) * padded_data_size_X] = in[x + y * data_size_X];
      }
    }
    
    // main convolution loop
#pragma omp parallel for
    for(int y = 0; y < data_size_Y; y++) { // the y coordinate of theoutput location we're focusing on
      for(int x = 0; x < 700; x += 100) { // the x coordinate of the output location we're focusing on
        __m128 sum_vec = _mm_setzero_ps();
        __m128 sum_vec2 = _mm_setzero_ps();
        __m128 sum_vec3 = _mm_setzero_ps();
        __m128 sum_vec4 = _mm_setzero_ps();
        __m128 sum_vec5 = _mm_setzero_ps();
        __m128 sum_vec6 = _mm_setzero_ps();
        __m128 sum_vec7 = _mm_setzero_ps();
        __m128 sum_vec8 = _mm_setzero_ps();
        __m128 sum_vec9 = _mm_setzero_ps();
        __m128 sum_vec10 = _mm_setzero_ps();
        __m128 sum_vec11 = _mm_setzero_ps();
        __m128 sum_vec12 = _mm_setzero_ps();
        __m128 sum_vec13 = _mm_setzero_ps();
        __m128 sum_vec14 = _mm_setzero_ps();
        __m128 sum_vec15 = _mm_setzero_ps();
        __m128 sum_vec16 = _mm_setzero_ps();
        __m128 sum_vec17 = _mm_setzero_ps();
        __m128 sum_vec18 = _mm_setzero_ps();
        __m128 sum_vec19 = _mm_setzero_ps();
        __m128 sum_vec20 = _mm_setzero_ps();
        __m128 sum_vec21 = _mm_setzero_ps();
        __m128 sum_vec22 = _mm_setzero_ps();
        __m128 sum_vec23 = _mm_setzero_ps();
        __m128 sum_vec24 = _mm_setzero_ps();
        __m128 sum_vec25 = _mm_setzero_ps();
        
        for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
          for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
            float *padded_in_Offset = padded_in + x + i + (y + j) *padded_data_size_X;
            __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
            
            __m128 input_vec = _mm_loadu_ps(padded_in_Offset);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
            __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset + 4);
            sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
            __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset + 8);
            sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
            __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset + 12);
            sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
            __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset + 16);
            sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
            __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset + 20);
            sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
            __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset + 24);
            sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
            __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset + 28);
            sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
            __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset + 32);
            sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
            __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset + 36);
            sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
            __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset + 40);
            sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
            __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset + 44);
            sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
            __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset + 48);
            sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
            __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset + 52);
            sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
            __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset + 56);
            sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
            __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset + 60);
            sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
            __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset + 64);
            sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
            __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset + 68);
            sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
            __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset + 72);
            sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
            __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset + 76);
            sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
            __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset + 80);
            sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
            __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset + 84);
            sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
            __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset + 88);
            sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
            __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset + 92);
            sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
            __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset + 96);
            sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
          }
        }
        float *outOffset = out + x + y * data_size_X;
        _mm_storeu_ps(outOffset, sum_vec);
        _mm_storeu_ps(outOffset + 4, sum_vec2);
        _mm_storeu_ps(outOffset + 8, sum_vec3);
        _mm_storeu_ps(outOffset + 12, sum_vec4);
        _mm_storeu_ps(outOffset + 16, sum_vec5);
        _mm_storeu_ps(outOffset + 20, sum_vec6);
        _mm_storeu_ps(outOffset + 24, sum_vec7);
        _mm_storeu_ps(outOffset + 28, sum_vec8);
        _mm_storeu_ps(outOffset + 32, sum_vec9);
        _mm_storeu_ps(outOffset + 36, sum_vec10);
        _mm_storeu_ps(outOffset + 40, sum_vec11);
        _mm_storeu_ps(outOffset + 44, sum_vec12);
        _mm_storeu_ps(outOffset + 48, sum_vec13);
        _mm_storeu_ps(outOffset + 52, sum_vec14);
        _mm_storeu_ps(outOffset + 56, sum_vec15);
        _mm_storeu_ps(outOffset + 60, sum_vec16);
        _mm_storeu_ps(outOffset + 64, sum_vec17);
        _mm_storeu_ps(outOffset + 68, sum_vec18);
        _mm_storeu_ps(outOffset + 72, sum_vec19);
        _mm_storeu_ps(outOffset + 76, sum_vec20);
        _mm_storeu_ps(outOffset + 80, sum_vec21);
        _mm_storeu_ps(outOffset + 84, sum_vec22);
        _mm_storeu_ps(outOffset + 88, sum_vec23);
        _mm_storeu_ps(outOffset + 92, sum_vec24);
        _mm_storeu_ps(outOffset + 96, sum_vec25);
			}
      
      // for sizes over 700
			int x = 700;
      __m128 sum_vec = _mm_setzero_ps();
      __m128 sum_vec2 = _mm_setzero_ps();
      __m128 sum_vec3 = _mm_setzero_ps();
      __m128 sum_vec4 = _mm_setzero_ps();
      __m128 sum_vec5 = _mm_setzero_ps();
      __m128 sum_vec6 = _mm_setzero_ps();
      __m128 sum_vec7 = _mm_setzero_ps();
      __m128 sum_vec8 = _mm_setzero_ps();
      __m128 sum_vec9 = _mm_setzero_ps();
      __m128 sum_vec10 = _mm_setzero_ps();
      __m128 sum_vec11 = _mm_setzero_ps();
      __m128 sum_vec12 = _mm_setzero_ps();
      __m128 sum_vec13 = _mm_setzero_ps();
      __m128 sum_vec14 = _mm_setzero_ps();
      __m128 sum_vec15 = _mm_setzero_ps();
      __m128 sum_vec16 = _mm_setzero_ps();
      __m128 sum_vec17 = _mm_setzero_ps();
      __m128 sum_vec18 = _mm_setzero_ps();
      __m128 sum_vec19 = _mm_setzero_ps();
      __m128 sum_vec20 = _mm_setzero_ps();
      __m128 sum_vec21 = _mm_setzero_ps();
      __m128 sum_vec22 = _mm_setzero_ps();
      __m128 sum_vec23 = _mm_setzero_ps();
      __m128 sum_vec24 = _mm_setzero_ps();
      __m128 sum_vec25 = _mm_setzero_ps();
      
      for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
        for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
          
          float *padded_in_Offset2 = padded_in + x + i + (y + j) *padded_data_size_X;
          __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
          
          __m128 input_vec = _mm_loadu_ps(padded_in_Offset2);
          sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
          __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset2 + 4);
          sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
          __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset2 + 8);
          sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
          __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset2 + 12);
          sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
          __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset2 + 16);
          sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
          __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset2 + 20);
          sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
          __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset2 + 24);
          sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
          __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset2 + 28);
          sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
          __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset2 + 32);
          sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
          __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset2 + 36);
          sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
          __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset2 + 40);
          sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
          __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset2 + 44);
          sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
          __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset2 + 48);
          sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
          __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset2 + 52);
          sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
          __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset2 + 56);
          sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
          __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset2 + 60);
          sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
          __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset2 + 64);
          sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
          __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset2 + 68);
          sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
          __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset2 + 72);
          sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
          __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset2 + 76);
          sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
          __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset2 + 80);
          sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
          __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset2 + 84);
          sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
          __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset2 + 88);
          sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
          __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset2 + 92);
          sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
          __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset2 + 96);
          sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
        }
      }
      
      float *padded_out_ptr = padded_out + x + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      _mm_storeu_ps(padded_out_ptr, sum_vec);
      _mm_storeu_ps(padded_out_ptr + 4, sum_vec2);
      _mm_storeu_ps(padded_out_ptr + 8, sum_vec3);
      _mm_storeu_ps(padded_out_ptr + 12, sum_vec4);
      _mm_storeu_ps(padded_out_ptr + 16, sum_vec5);
      _mm_storeu_ps(padded_out_ptr + 20, sum_vec6);
      _mm_storeu_ps(padded_out_ptr + 24, sum_vec7);
      _mm_storeu_ps(padded_out_ptr + 28, sum_vec8);
      _mm_storeu_ps(padded_out_ptr + 32, sum_vec9);
      _mm_storeu_ps(padded_out_ptr + 36, sum_vec10);
      _mm_storeu_ps(padded_out_ptr + 40, sum_vec11);
      _mm_storeu_ps(padded_out_ptr + 44, sum_vec12);
      _mm_storeu_ps(padded_out_ptr + 48, sum_vec13);
      _mm_storeu_ps(padded_out_ptr + 52, sum_vec14);
      _mm_storeu_ps(padded_out_ptr + 56, sum_vec15);
      _mm_storeu_ps(padded_out_ptr + 60, sum_vec16);
      _mm_storeu_ps(padded_out_ptr + 64, sum_vec17);
      _mm_storeu_ps(padded_out_ptr + 68, sum_vec18);
      _mm_storeu_ps(padded_out_ptr + 72, sum_vec19);
      _mm_storeu_ps(padded_out_ptr + 76, sum_vec20);
      _mm_storeu_ps(padded_out_ptr + 80, sum_vec21);
      _mm_storeu_ps(padded_out_ptr + 84, sum_vec22);
      _mm_storeu_ps(padded_out_ptr + 88, sum_vec23);
      _mm_storeu_ps(padded_out_ptr + 92, sum_vec24);
      _mm_storeu_ps(padded_out_ptr + 96, sum_vec25);
    }
    
    int y_MUL_data_size_X;
    int y_PLUS_kern_MUL_pad;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      y_MUL_data_size_X = y * data_size_X;
      y_PLUS_kern_MUL_pad = (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 700; x < data_size_X; x++) {
        out[x + y_MUL_data_size_X] =	padded_out[x + kern_cent_X + y_PLUS_kern_MUL_pad];
      }
    }
    
    /*****************************/
    /*    Greater than 600       */
    /*****************************/
  } else if (data_size_X > 600) {
    static float padded_in[PADDEDMATRIXSIZE]__attribute__((aligned(16))); // 1204 * 1204  
    static int trials;
    trials++;
    
    if (trials > 1) { 
      // fill the preinput matrix with zeros
      int padded_size = (padded_data_size_X)* (padded_data_size_Y); // maybe remove or lower the padded_size bounds
      int padded_size_block = padded_size/100*100;
#pragma omp parallel for
      for (int z = 400; z < padded_size_block; z += 100) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 4 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 8 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 12 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 16 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 20 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 24 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 28 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 32 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 36 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 40 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 44 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 48 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 52 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 56 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 60 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 64 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 68 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 72 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 76 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 80 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 84 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 88 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 92 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 96 + z, _mm_setzero_ps());
      }
      for (int z = padded_size_block; z < padded_size; z += 4) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
      }
    }

    // Insert input matrix into padded matrix
    int x_boundary = data_size_X/32*32;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      float *in_add_offset_orig = in + y * data_size_X;
      float *padded_in_offset_3_orig = padded_in + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 0; x < x_boundary; x += 32) {
        float *in_add_offset = in_add_offset_orig + x;
        float *padded_in_offset_3 = padded_in_offset_3_orig + x;
        _mm_storeu_ps(padded_in_offset_3, _mm_loadu_ps(in_add_offset));
        _mm_storeu_ps(padded_in_offset_3 + 4, _mm_loadu_ps(in_add_offset + 4));
        _mm_storeu_ps(padded_in_offset_3 + 8, _mm_loadu_ps(in_add_offset + 8));
        _mm_storeu_ps(padded_in_offset_3 + 12, _mm_loadu_ps(in_add_offset + 12));
        _mm_storeu_ps(padded_in_offset_3 + 16, _mm_loadu_ps(in_add_offset + 16));
        _mm_storeu_ps(padded_in_offset_3 + 20, _mm_loadu_ps(in_add_offset + 20));
        _mm_storeu_ps(padded_in_offset_3 + 24, _mm_loadu_ps(in_add_offset + 24));
        _mm_storeu_ps(padded_in_offset_3 + 28, _mm_loadu_ps(in_add_offset + 28));
      }
      for (int x = x_boundary; x < data_size_X; x++) {
        padded_in[(x + kern_cent_X) + (y + kern_cent_Y) * padded_data_size_X] = in[x + y * data_size_X];
      }
    }
    
    // main convolution loop
#pragma omp parallel for
    for(int y = 0; y < data_size_Y; y++) { // the y coordinate of theoutput location we're focusing on
      // because we know we have a minimum size of 400
      for(int x = 0; x < 600; x += 100) { // the x coordinate of the output location we're focusing on
        __m128 sum_vec = _mm_setzero_ps();
        __m128 sum_vec2 = _mm_setzero_ps();
        __m128 sum_vec3 = _mm_setzero_ps();
        __m128 sum_vec4 = _mm_setzero_ps();
        __m128 sum_vec5 = _mm_setzero_ps();
        __m128 sum_vec6 = _mm_setzero_ps();
        __m128 sum_vec7 = _mm_setzero_ps();
        __m128 sum_vec8 = _mm_setzero_ps();
        __m128 sum_vec9 = _mm_setzero_ps();
        __m128 sum_vec10 = _mm_setzero_ps();
        __m128 sum_vec11 = _mm_setzero_ps();
        __m128 sum_vec12 = _mm_setzero_ps();
        __m128 sum_vec13 = _mm_setzero_ps();
        __m128 sum_vec14 = _mm_setzero_ps();
        __m128 sum_vec15 = _mm_setzero_ps();
        __m128 sum_vec16 = _mm_setzero_ps();
        __m128 sum_vec17 = _mm_setzero_ps();
        __m128 sum_vec18 = _mm_setzero_ps();
        __m128 sum_vec19 = _mm_setzero_ps();
        __m128 sum_vec20 = _mm_setzero_ps();
        __m128 sum_vec21 = _mm_setzero_ps();
        __m128 sum_vec22 = _mm_setzero_ps();
        __m128 sum_vec23 = _mm_setzero_ps();
        __m128 sum_vec24 = _mm_setzero_ps();
        __m128 sum_vec25 = _mm_setzero_ps();
        
        for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
          for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
            float *padded_in_Offset = padded_in + x + i + (y + j) *padded_data_size_X;
            __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
            
            __m128 input_vec = _mm_loadu_ps(padded_in_Offset);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
            __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset + 4);
            sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
            __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset + 8);
            sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
            __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset + 12);
            sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
            __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset + 16);
            sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
            __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset + 20);
            sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
            __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset + 24);
            sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
            __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset + 28);
            sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
            __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset + 32);
            sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
            __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset + 36);
            sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
            __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset + 40);
            sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
            __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset + 44);
            sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
            __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset + 48);
            sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
            __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset + 52);
            sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
            __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset + 56);
            sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
            __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset + 60);
            sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
            __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset + 64);
            sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
            __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset + 68);
            sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
            __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset + 72);
            sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
            __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset + 76);
            sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
            __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset + 80);
            sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
            __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset + 84);
            sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
            __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset + 88);
            sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
            __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset + 92);
            sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
            __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset + 96);
            sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
          }
        }
        float *outOffset = out + x + y * data_size_X;
        _mm_storeu_ps(outOffset, sum_vec);
        _mm_storeu_ps(outOffset + 4, sum_vec2);
        _mm_storeu_ps(outOffset + 8, sum_vec3);
        _mm_storeu_ps(outOffset + 12, sum_vec4);
        _mm_storeu_ps(outOffset + 16, sum_vec5);
        _mm_storeu_ps(outOffset + 20, sum_vec6);
        _mm_storeu_ps(outOffset + 24, sum_vec7);
        _mm_storeu_ps(outOffset + 28, sum_vec8);
        _mm_storeu_ps(outOffset + 32, sum_vec9);
        _mm_storeu_ps(outOffset + 36, sum_vec10);
        _mm_storeu_ps(outOffset + 40, sum_vec11);
        _mm_storeu_ps(outOffset + 44, sum_vec12);
        _mm_storeu_ps(outOffset + 48, sum_vec13);
        _mm_storeu_ps(outOffset + 52, sum_vec14);
        _mm_storeu_ps(outOffset + 56, sum_vec15);
        _mm_storeu_ps(outOffset + 60, sum_vec16);
        _mm_storeu_ps(outOffset + 64, sum_vec17);
        _mm_storeu_ps(outOffset + 68, sum_vec18);
        _mm_storeu_ps(outOffset + 72, sum_vec19);
        _mm_storeu_ps(outOffset + 76, sum_vec20);
        _mm_storeu_ps(outOffset + 80, sum_vec21);
        _mm_storeu_ps(outOffset + 84, sum_vec22);
        _mm_storeu_ps(outOffset + 88, sum_vec23);
        _mm_storeu_ps(outOffset + 92, sum_vec24);
        _mm_storeu_ps(outOffset + 96, sum_vec25);
			}
      
      // for sizes over 600
			int x = 600;
      __m128 sum_vec = _mm_setzero_ps();
      __m128 sum_vec2 = _mm_setzero_ps();
      __m128 sum_vec3 = _mm_setzero_ps();
      __m128 sum_vec4 = _mm_setzero_ps();
      __m128 sum_vec5 = _mm_setzero_ps();
      __m128 sum_vec6 = _mm_setzero_ps();
      __m128 sum_vec7 = _mm_setzero_ps();
      __m128 sum_vec8 = _mm_setzero_ps();
      __m128 sum_vec9 = _mm_setzero_ps();
      __m128 sum_vec10 = _mm_setzero_ps();
      __m128 sum_vec11 = _mm_setzero_ps();
      __m128 sum_vec12 = _mm_setzero_ps();
      __m128 sum_vec13 = _mm_setzero_ps();
      __m128 sum_vec14 = _mm_setzero_ps();
      __m128 sum_vec15 = _mm_setzero_ps();
      __m128 sum_vec16 = _mm_setzero_ps();
      __m128 sum_vec17 = _mm_setzero_ps();
      __m128 sum_vec18 = _mm_setzero_ps();
      __m128 sum_vec19 = _mm_setzero_ps();
      __m128 sum_vec20 = _mm_setzero_ps();
      __m128 sum_vec21 = _mm_setzero_ps();
      __m128 sum_vec22 = _mm_setzero_ps();
      __m128 sum_vec23 = _mm_setzero_ps();
      __m128 sum_vec24 = _mm_setzero_ps();
      __m128 sum_vec25 = _mm_setzero_ps();
      
      for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
        for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
          
          float *padded_in_Offset2 = padded_in + x + i + (y + j) *padded_data_size_X;
          __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
          
          __m128 input_vec = _mm_loadu_ps(padded_in_Offset2);
          sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
          __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset2 + 4);
          sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
          __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset2 + 8);
          sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
          __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset2 + 12);
          sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
          __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset2 + 16);
          sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
          __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset2 + 20);
          sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
          __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset2 + 24);
          sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
          __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset2 + 28);
          sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
          __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset2 + 32);
          sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
          __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset2 + 36);
          sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
          __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset2 + 40);
          sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
          __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset2 + 44);
          sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
          __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset2 + 48);
          sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
          __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset2 + 52);
          sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
          __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset2 + 56);
          sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
          __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset2 + 60);
          sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
          __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset2 + 64);
          sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
          __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset2 + 68);
          sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
          __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset2 + 72);
          sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
          __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset2 + 76);
          sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
          __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset2 + 80);
          sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
          __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset2 + 84);
          sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
          __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset2 + 88);
          sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
          __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset2 + 92);
          sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
          __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset2 + 96);
          sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
        }
      }
      
      float *padded_out_ptr = padded_out + x + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      _mm_storeu_ps(padded_out_ptr, sum_vec);
      _mm_storeu_ps(padded_out_ptr + 4, sum_vec2);
      _mm_storeu_ps(padded_out_ptr + 8, sum_vec3);
      _mm_storeu_ps(padded_out_ptr + 12, sum_vec4);
      _mm_storeu_ps(padded_out_ptr + 16, sum_vec5);
      _mm_storeu_ps(padded_out_ptr + 20, sum_vec6);
      _mm_storeu_ps(padded_out_ptr + 24, sum_vec7);
      _mm_storeu_ps(padded_out_ptr + 28, sum_vec8);
      _mm_storeu_ps(padded_out_ptr + 32, sum_vec9);
      _mm_storeu_ps(padded_out_ptr + 36, sum_vec10);
      _mm_storeu_ps(padded_out_ptr + 40, sum_vec11);
      _mm_storeu_ps(padded_out_ptr + 44, sum_vec12);
      _mm_storeu_ps(padded_out_ptr + 48, sum_vec13);
      _mm_storeu_ps(padded_out_ptr + 52, sum_vec14);
      _mm_storeu_ps(padded_out_ptr + 56, sum_vec15);
      _mm_storeu_ps(padded_out_ptr + 60, sum_vec16);
      _mm_storeu_ps(padded_out_ptr + 64, sum_vec17);
      _mm_storeu_ps(padded_out_ptr + 68, sum_vec18);
      _mm_storeu_ps(padded_out_ptr + 72, sum_vec19);
      _mm_storeu_ps(padded_out_ptr + 76, sum_vec20);
      _mm_storeu_ps(padded_out_ptr + 80, sum_vec21);
      _mm_storeu_ps(padded_out_ptr + 84, sum_vec22);
      _mm_storeu_ps(padded_out_ptr + 88, sum_vec23);
      _mm_storeu_ps(padded_out_ptr + 92, sum_vec24);
      _mm_storeu_ps(padded_out_ptr + 96, sum_vec25);
    }
    
    int y_MUL_data_size_X;
    int y_PLUS_kern_MUL_pad;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      y_MUL_data_size_X = y * data_size_X;
      y_PLUS_kern_MUL_pad = (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 600; x < data_size_X; x++) {
        out[x + y_MUL_data_size_X] =	padded_out[x + kern_cent_X + y_PLUS_kern_MUL_pad];
      }
    }
  
    /*****************************/
    /*    Greater than 500       */
    /*****************************/
  } else if (data_size_X > 500) {
		static float padded_in[PADDEDMATRIXSIZE]__attribute__((aligned(16))); // 1204 * 1204
		static int trials;
		trials++;
    
		if (trials > 1) {
      // fill the preinput matrix with zeros
      int padded_size = (padded_data_size_X)* (padded_data_size_Y); // maybe remove or lower the padded_size bounds
      int padded_size_block = padded_size/100*100;
#pragma omp parallel for
      for (int z = 400; z < padded_size_block; z += 100) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 4 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 8 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 12 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 16 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 20 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 24 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 28 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 32 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 36 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 40 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 44 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 48 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 52 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 56 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 60 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 64 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 68 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 72 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 76 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 80 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 84 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 88 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 92 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 96 + z, _mm_setzero_ps());
      }
      for (int z = padded_size_block; z < padded_size; z += 4) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
      }
		}
 
    // Insert input matrix into padded matrix
    int x_boundary = data_size_X/32*32;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      float *in_add_offset_orig = in + y * data_size_X;
      float *padded_in_offset_3_orig = padded_in + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 0; x < x_boundary; x += 32) {
        float *in_add_offset = in_add_offset_orig + x;
        float *padded_in_offset_3 = padded_in_offset_3_orig + x;
        _mm_storeu_ps(padded_in_offset_3, _mm_loadu_ps(in_add_offset));
        _mm_storeu_ps(padded_in_offset_3 + 4, _mm_loadu_ps(in_add_offset + 4));
        _mm_storeu_ps(padded_in_offset_3 + 8, _mm_loadu_ps(in_add_offset + 8));
        _mm_storeu_ps(padded_in_offset_3 + 12, _mm_loadu_ps(in_add_offset + 12));
        _mm_storeu_ps(padded_in_offset_3 + 16, _mm_loadu_ps(in_add_offset + 16));
        _mm_storeu_ps(padded_in_offset_3 + 20, _mm_loadu_ps(in_add_offset + 20));
        _mm_storeu_ps(padded_in_offset_3 + 24, _mm_loadu_ps(in_add_offset + 24));
        _mm_storeu_ps(padded_in_offset_3 + 28, _mm_loadu_ps(in_add_offset + 28));
      }
      for (int x = x_boundary; x < data_size_X; x++) {
        padded_in[(x + kern_cent_X) + (y + kern_cent_Y) * padded_data_size_X] = in[x + y * data_size_X];
      }
    }
    
    // main convolution loop
#pragma omp parallel for
    for(int y = 0; y < data_size_Y; y++) { // the y coordinate of theoutput location we're focusing on
      for(int x = 0; x < 500; x += 100) { // the x coordinate of the output location we're focusing on
        __m128 sum_vec = _mm_setzero_ps();
        __m128 sum_vec2 = _mm_setzero_ps();
        __m128 sum_vec3 = _mm_setzero_ps();
        __m128 sum_vec4 = _mm_setzero_ps();
        __m128 sum_vec5 = _mm_setzero_ps();
        __m128 sum_vec6 = _mm_setzero_ps();
        __m128 sum_vec7 = _mm_setzero_ps();
        __m128 sum_vec8 = _mm_setzero_ps();
        __m128 sum_vec9 = _mm_setzero_ps();
        __m128 sum_vec10 = _mm_setzero_ps();
        __m128 sum_vec11 = _mm_setzero_ps();
        __m128 sum_vec12 = _mm_setzero_ps();
        __m128 sum_vec13 = _mm_setzero_ps();
        __m128 sum_vec14 = _mm_setzero_ps();
        __m128 sum_vec15 = _mm_setzero_ps();
        __m128 sum_vec16 = _mm_setzero_ps();
        __m128 sum_vec17 = _mm_setzero_ps();
        __m128 sum_vec18 = _mm_setzero_ps();
        __m128 sum_vec19 = _mm_setzero_ps();
        __m128 sum_vec20 = _mm_setzero_ps();
        __m128 sum_vec21 = _mm_setzero_ps();
        __m128 sum_vec22 = _mm_setzero_ps();
        __m128 sum_vec23 = _mm_setzero_ps();
        __m128 sum_vec24 = _mm_setzero_ps();
        __m128 sum_vec25 = _mm_setzero_ps();
        
        for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
          for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
            float *padded_in_Offset = padded_in + x + i + (y + j) *padded_data_size_X;
            __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
            
            __m128 input_vec = _mm_loadu_ps(padded_in_Offset);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
            __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset + 4);
            sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
            __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset + 8);
            sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
            __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset + 12);
            sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
            __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset + 16);
            sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
            __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset + 20);
            sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
            __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset + 24);
            sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
            __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset + 28);
            sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
            __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset + 32);
            sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
            __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset + 36);
            sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
            __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset + 40);
            sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
            __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset + 44);
            sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
            __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset + 48);
            sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
            __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset + 52);
            sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
            __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset + 56);
            sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
            __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset + 60);
            sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
            __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset + 64);
            sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
            __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset + 68);
            sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
            __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset + 72);
            sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
            __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset + 76);
            sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
            __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset + 80);
            sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
            __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset + 84);
            sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
            __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset + 88);
            sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
            __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset + 92);
            sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
            __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset + 96);
            sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
          }
        }
        float *outOffset = out + x + y * data_size_X;
        _mm_storeu_ps(outOffset, sum_vec);
        _mm_storeu_ps(outOffset + 4, sum_vec2);
        _mm_storeu_ps(outOffset + 8, sum_vec3);
        _mm_storeu_ps(outOffset + 12, sum_vec4);
        _mm_storeu_ps(outOffset + 16, sum_vec5);
        _mm_storeu_ps(outOffset + 20, sum_vec6);
        _mm_storeu_ps(outOffset + 24, sum_vec7);
        _mm_storeu_ps(outOffset + 28, sum_vec8);
        _mm_storeu_ps(outOffset + 32, sum_vec9);
        _mm_storeu_ps(outOffset + 36, sum_vec10);
        _mm_storeu_ps(outOffset + 40, sum_vec11);
        _mm_storeu_ps(outOffset + 44, sum_vec12);
        _mm_storeu_ps(outOffset + 48, sum_vec13);
        _mm_storeu_ps(outOffset + 52, sum_vec14);
        _mm_storeu_ps(outOffset + 56, sum_vec15);
        _mm_storeu_ps(outOffset + 60, sum_vec16);
        _mm_storeu_ps(outOffset + 64, sum_vec17);
        _mm_storeu_ps(outOffset + 68, sum_vec18);
        _mm_storeu_ps(outOffset + 72, sum_vec19);
        _mm_storeu_ps(outOffset + 76, sum_vec20);
        _mm_storeu_ps(outOffset + 80, sum_vec21);
        _mm_storeu_ps(outOffset + 84, sum_vec22);
        _mm_storeu_ps(outOffset + 88, sum_vec23);
        _mm_storeu_ps(outOffset + 92, sum_vec24);
        _mm_storeu_ps(outOffset + 96, sum_vec25);
			}
      
      // for sizes over 500
			int x = 500;
      __m128 sum_vec = _mm_setzero_ps();
      __m128 sum_vec2 = _mm_setzero_ps();
      __m128 sum_vec3 = _mm_setzero_ps();
      __m128 sum_vec4 = _mm_setzero_ps();
      __m128 sum_vec5 = _mm_setzero_ps();
      __m128 sum_vec6 = _mm_setzero_ps();
      __m128 sum_vec7 = _mm_setzero_ps();
      __m128 sum_vec8 = _mm_setzero_ps();
      __m128 sum_vec9 = _mm_setzero_ps();
      __m128 sum_vec10 = _mm_setzero_ps();
      __m128 sum_vec11 = _mm_setzero_ps();
      __m128 sum_vec12 = _mm_setzero_ps();
      __m128 sum_vec13 = _mm_setzero_ps();
      __m128 sum_vec14 = _mm_setzero_ps();
      __m128 sum_vec15 = _mm_setzero_ps();
      __m128 sum_vec16 = _mm_setzero_ps();
      __m128 sum_vec17 = _mm_setzero_ps();
      __m128 sum_vec18 = _mm_setzero_ps();
      __m128 sum_vec19 = _mm_setzero_ps();
      __m128 sum_vec20 = _mm_setzero_ps();
      __m128 sum_vec21 = _mm_setzero_ps();
      __m128 sum_vec22 = _mm_setzero_ps();
      __m128 sum_vec23 = _mm_setzero_ps();
      __m128 sum_vec24 = _mm_setzero_ps();
      __m128 sum_vec25 = _mm_setzero_ps();
      
      for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
        for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
          
          float *padded_in_Offset2 = padded_in + x + i + (y + j) *padded_data_size_X;
          __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
          
          __m128 input_vec = _mm_loadu_ps(padded_in_Offset2);
          sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
          __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset2 + 4);
          sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
          __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset2 + 8);
          sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
          __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset2 + 12);
          sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
          __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset2 + 16);
          sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
          __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset2 + 20);
          sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
          __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset2 + 24);
          sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
          __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset2 + 28);
          sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
          __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset2 + 32);
          sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
          __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset2 + 36);
          sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
          __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset2 + 40);
          sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
          __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset2 + 44);
          sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
          __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset2 + 48);
          sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
          __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset2 + 52);
          sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
          __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset2 + 56);
          sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
          __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset2 + 60);
          sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
          __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset2 + 64);
          sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
          __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset2 + 68);
          sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
          __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset2 + 72);
          sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
          __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset2 + 76);
          sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
          __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset2 + 80);
          sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
          __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset2 + 84);
          sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
          __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset2 + 88);
          sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
          __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset2 + 92);
          sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
          __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset2 + 96);
          sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
        }
      }
      
      float *padded_out_ptr = padded_out + x + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      _mm_storeu_ps(padded_out_ptr, sum_vec);
      _mm_storeu_ps(padded_out_ptr + 4, sum_vec2);
      _mm_storeu_ps(padded_out_ptr + 8, sum_vec3);
      _mm_storeu_ps(padded_out_ptr + 12, sum_vec4);
      _mm_storeu_ps(padded_out_ptr + 16, sum_vec5);
      _mm_storeu_ps(padded_out_ptr + 20, sum_vec6);
      _mm_storeu_ps(padded_out_ptr + 24, sum_vec7);
      _mm_storeu_ps(padded_out_ptr + 28, sum_vec8);
      _mm_storeu_ps(padded_out_ptr + 32, sum_vec9);
      _mm_storeu_ps(padded_out_ptr + 36, sum_vec10);
      _mm_storeu_ps(padded_out_ptr + 40, sum_vec11);
      _mm_storeu_ps(padded_out_ptr + 44, sum_vec12);
      _mm_storeu_ps(padded_out_ptr + 48, sum_vec13);
      _mm_storeu_ps(padded_out_ptr + 52, sum_vec14);
      _mm_storeu_ps(padded_out_ptr + 56, sum_vec15);
      _mm_storeu_ps(padded_out_ptr + 60, sum_vec16);
      _mm_storeu_ps(padded_out_ptr + 64, sum_vec17);
      _mm_storeu_ps(padded_out_ptr + 68, sum_vec18);
      _mm_storeu_ps(padded_out_ptr + 72, sum_vec19);
      _mm_storeu_ps(padded_out_ptr + 76, sum_vec20);
      _mm_storeu_ps(padded_out_ptr + 80, sum_vec21);
      _mm_storeu_ps(padded_out_ptr + 84, sum_vec22);
      _mm_storeu_ps(padded_out_ptr + 88, sum_vec23);
      _mm_storeu_ps(padded_out_ptr + 92, sum_vec24);
      _mm_storeu_ps(padded_out_ptr + 96, sum_vec25);
    }
    
    int y_MUL_data_size_X;
    int y_PLUS_kern_MUL_pad;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      y_MUL_data_size_X = y * data_size_X;
      y_PLUS_kern_MUL_pad = (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 500; x < data_size_X; x++) {
        out[x + y_MUL_data_size_X] =	padded_out[x + kern_cent_X + y_PLUS_kern_MUL_pad];
      }
    }
    
    /*****************************/
    /*    Greater than 400       */
    /*****************************/
  } else {
    static float padded_in[PADDEDMATRIXSIZE]__attribute__((aligned(16))); // 1204 * 1204    
    static int trials;
    trials++;
    
    if (trials > 1) { 
      // fill the preinput matrix with zeros
      int padded_size = (padded_data_size_X)* (padded_data_size_Y); // maybe remove or lower the padded_size bounds
      int padded_size_block = padded_size/100*100;
#pragma omp parallel for
      for (int z = 400; z < padded_size_block; z += 100) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 4 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 8 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 12 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 16 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 20 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 24 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 28 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 32 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 36 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 40 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 44 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 48 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 52 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 56 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 60 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 64 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 68 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 72 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 76 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 80 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 84 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 88 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 92 + z, _mm_setzero_ps());
        _mm_store_ps(padded_in + 96 + z, _mm_setzero_ps());
      }
      for (int z = padded_size_block; z < padded_size; z += 4) {
        _mm_store_ps(padded_in + z, _mm_setzero_ps());
      }
    }

    // Insert input matrix into padded matrix
    int x_boundary = data_size_X/32*32;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      float *in_add_offset_orig = in + y * data_size_X;
      float *padded_in_offset_3_orig = padded_in + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 0; x < x_boundary; x += 32) {
        float *in_add_offset = in_add_offset_orig + x;
        float *padded_in_offset_3 = padded_in_offset_3_orig + x;
        _mm_storeu_ps(padded_in_offset_3, _mm_loadu_ps(in_add_offset));
        _mm_storeu_ps(padded_in_offset_3 + 4, _mm_loadu_ps(in_add_offset + 4));
        _mm_storeu_ps(padded_in_offset_3 + 8, _mm_loadu_ps(in_add_offset + 8));
        _mm_storeu_ps(padded_in_offset_3 + 12, _mm_loadu_ps(in_add_offset + 12));
        _mm_storeu_ps(padded_in_offset_3 + 16, _mm_loadu_ps(in_add_offset + 16));
        _mm_storeu_ps(padded_in_offset_3 + 20, _mm_loadu_ps(in_add_offset + 20));
        _mm_storeu_ps(padded_in_offset_3 + 24, _mm_loadu_ps(in_add_offset + 24));
        _mm_storeu_ps(padded_in_offset_3 + 28, _mm_loadu_ps(in_add_offset + 28));
      }
      for (int x = x_boundary; x < data_size_X; x++) {
        padded_in[(x + kern_cent_X) + (y + kern_cent_Y) * padded_data_size_X] = in[x + y * data_size_X];
      }
    }
    
    // main convolution loop
#pragma omp parallel for
    for(int y = 0; y < data_size_Y; y++) { // the y coordinate of theoutput location we're focusing on
      for(int x = 0; x < 400; x += 100) { // the x coordinate of the output location we're focusing on
        __m128 sum_vec = _mm_setzero_ps();
        __m128 sum_vec2 = _mm_setzero_ps();
        __m128 sum_vec3 = _mm_setzero_ps();
        __m128 sum_vec4 = _mm_setzero_ps();
        __m128 sum_vec5 = _mm_setzero_ps();
        __m128 sum_vec6 = _mm_setzero_ps();
        __m128 sum_vec7 = _mm_setzero_ps();
        __m128 sum_vec8 = _mm_setzero_ps();
        __m128 sum_vec9 = _mm_setzero_ps();
        __m128 sum_vec10 = _mm_setzero_ps();
        __m128 sum_vec11 = _mm_setzero_ps();
        __m128 sum_vec12 = _mm_setzero_ps();
        __m128 sum_vec13 = _mm_setzero_ps();
        __m128 sum_vec14 = _mm_setzero_ps();
        __m128 sum_vec15 = _mm_setzero_ps();
        __m128 sum_vec16 = _mm_setzero_ps();
        __m128 sum_vec17 = _mm_setzero_ps();
        __m128 sum_vec18 = _mm_setzero_ps();
        __m128 sum_vec19 = _mm_setzero_ps();
        __m128 sum_vec20 = _mm_setzero_ps();
        __m128 sum_vec21 = _mm_setzero_ps();
        __m128 sum_vec22 = _mm_setzero_ps();
        __m128 sum_vec23 = _mm_setzero_ps();
        __m128 sum_vec24 = _mm_setzero_ps();
        __m128 sum_vec25 = _mm_setzero_ps();
        
        for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
          for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
            float *padded_in_Offset = padded_in + x + i + (y + j) *padded_data_size_X;
            __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
            
            __m128 input_vec = _mm_loadu_ps(padded_in_Offset);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
            __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset + 4);
            sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
            __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset + 8);
            sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
            __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset + 12);
            sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
            __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset + 16);
            sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
            __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset + 20);
            sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
            __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset + 24);
            sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
            __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset + 28);
            sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
            __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset + 32);
            sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
            __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset + 36);
            sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
            __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset + 40);
            sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
            __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset + 44);
            sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
            __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset + 48);
            sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
            __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset + 52);
            sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
            __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset + 56);
            sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
            __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset + 60);
            sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
            __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset + 64);
            sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
            __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset + 68);
            sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
            __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset + 72);
            sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
            __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset + 76);
            sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
            __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset + 80);
            sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
            __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset + 84);
            sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
            __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset + 88);
            sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
            __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset + 92);
            sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));
            __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset + 96);
            sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));
          }
        }
        float *outOffset = out + x + y * data_size_X;
        _mm_storeu_ps(outOffset, sum_vec);
        _mm_storeu_ps(outOffset + 4, sum_vec2);
        _mm_storeu_ps(outOffset + 8, sum_vec3);
        _mm_storeu_ps(outOffset + 12, sum_vec4);
        _mm_storeu_ps(outOffset + 16, sum_vec5);
        _mm_storeu_ps(outOffset + 20, sum_vec6);
        _mm_storeu_ps(outOffset + 24, sum_vec7);
        _mm_storeu_ps(outOffset + 28, sum_vec8);
        _mm_storeu_ps(outOffset + 32, sum_vec9);
        _mm_storeu_ps(outOffset + 36, sum_vec10);
        _mm_storeu_ps(outOffset + 40, sum_vec11);
        _mm_storeu_ps(outOffset + 44, sum_vec12);
        _mm_storeu_ps(outOffset + 48, sum_vec13);
        _mm_storeu_ps(outOffset + 52, sum_vec14);
        _mm_storeu_ps(outOffset + 56, sum_vec15);
        _mm_storeu_ps(outOffset + 60, sum_vec16);
        _mm_storeu_ps(outOffset + 64, sum_vec17);
        _mm_storeu_ps(outOffset + 68, sum_vec18);
        _mm_storeu_ps(outOffset + 72, sum_vec19);
        _mm_storeu_ps(outOffset + 76, sum_vec20);
        _mm_storeu_ps(outOffset + 80, sum_vec21);
        _mm_storeu_ps(outOffset + 84, sum_vec22);
        _mm_storeu_ps(outOffset + 88, sum_vec23);
        _mm_storeu_ps(outOffset + 92, sum_vec24);
        _mm_storeu_ps(outOffset + 96, sum_vec25);
			}
      
      // for sizes over 400
      //for(int x = 400; x < data_size_X; x += 100) { // the x coordinate of the output location we're focusing on
			int x = 400;
      __m128 sum_vec = _mm_setzero_ps();
      __m128 sum_vec2 = _mm_setzero_ps();
      __m128 sum_vec3 = _mm_setzero_ps();
      __m128 sum_vec4 = _mm_setzero_ps();
      __m128 sum_vec5 = _mm_setzero_ps();
      __m128 sum_vec6 = _mm_setzero_ps();
      __m128 sum_vec7 = _mm_setzero_ps();
      __m128 sum_vec8 = _mm_setzero_ps();
      __m128 sum_vec9 = _mm_setzero_ps();
      __m128 sum_vec10 = _mm_setzero_ps();
      __m128 sum_vec11 = _mm_setzero_ps();
      __m128 sum_vec12 = _mm_setzero_ps();
      __m128 sum_vec13 = _mm_setzero_ps();
      __m128 sum_vec14 = _mm_setzero_ps();
      __m128 sum_vec15 = _mm_setzero_ps();
      __m128 sum_vec16 = _mm_setzero_ps();
      __m128 sum_vec17 = _mm_setzero_ps();
      __m128 sum_vec18 = _mm_setzero_ps();
      __m128 sum_vec19 = _mm_setzero_ps();
      __m128 sum_vec20 = _mm_setzero_ps();
      __m128 sum_vec21 = _mm_setzero_ps();
      __m128 sum_vec22 = _mm_setzero_ps();
      __m128 sum_vec23 = _mm_setzero_ps();
      __m128 sum_vec24 = _mm_setzero_ps();
      __m128 sum_vec25 = _mm_setzero_ps();
      
      for(int j = 0; j < kernel_y; j++) { // kernel y coordinate
        for(int i = 0; i < kernel_x; i++) { // kernel x coordinate
          
          float *padded_in_Offset2 = padded_in + x + i + (y + j) *padded_data_size_X;
          __m128 kern_value = _mm_load1_ps(flip_kernel + i + j * kernel_x); // load 1 kernal value into 4 places in vector
          
          __m128 input_vec = _mm_loadu_ps(padded_in_Offset2);
          sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, kern_value));
          __m128 input_vec2 = _mm_loadu_ps(padded_in_Offset2 + 4);
          sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(input_vec2, kern_value));
          __m128 input_vec3 = _mm_loadu_ps(padded_in_Offset2 + 8);
          sum_vec3 = _mm_add_ps(sum_vec3, _mm_mul_ps(input_vec3, kern_value));
          __m128 input_vec4 = _mm_loadu_ps(padded_in_Offset2 + 12);
          sum_vec4 = _mm_add_ps(sum_vec4, _mm_mul_ps(input_vec4, kern_value));
          __m128 input_vec5 = _mm_loadu_ps(padded_in_Offset2 + 16);
          sum_vec5 = _mm_add_ps(sum_vec5, _mm_mul_ps(input_vec5, kern_value));
          __m128 input_vec6 = _mm_loadu_ps(padded_in_Offset2 + 20);
          sum_vec6 = _mm_add_ps(sum_vec6, _mm_mul_ps(input_vec6, kern_value));
          __m128 input_vec7 = _mm_loadu_ps(padded_in_Offset2 + 24);
          sum_vec7 = _mm_add_ps(sum_vec7, _mm_mul_ps(input_vec7, kern_value));
          __m128 input_vec8 = _mm_loadu_ps(padded_in_Offset2 + 28);
          sum_vec8 = _mm_add_ps(sum_vec8, _mm_mul_ps(input_vec8, kern_value));
          __m128 input_vec9 = _mm_loadu_ps(padded_in_Offset2 + 32);
          sum_vec9 = _mm_add_ps(sum_vec9, _mm_mul_ps(input_vec9, kern_value));
          __m128 input_vec10 = _mm_loadu_ps(padded_in_Offset2 + 36);
          sum_vec10 = _mm_add_ps(sum_vec10, _mm_mul_ps(input_vec10, kern_value));
          __m128 input_vec11 = _mm_loadu_ps(padded_in_Offset2 + 40);
          sum_vec11 = _mm_add_ps(sum_vec11, _mm_mul_ps(input_vec11, kern_value));
          __m128 input_vec12 = _mm_loadu_ps(padded_in_Offset2 + 44);
          sum_vec12 = _mm_add_ps(sum_vec12, _mm_mul_ps(input_vec12, kern_value));
          __m128 input_vec13 = _mm_loadu_ps(padded_in_Offset2 + 48);
          sum_vec13 = _mm_add_ps(sum_vec13, _mm_mul_ps(input_vec13, kern_value));
          __m128 input_vec14 = _mm_loadu_ps(padded_in_Offset2 + 52);
          sum_vec14 = _mm_add_ps(sum_vec14, _mm_mul_ps(input_vec14, kern_value));
          __m128 input_vec15 = _mm_loadu_ps(padded_in_Offset2 + 56);
          sum_vec15 = _mm_add_ps(sum_vec15, _mm_mul_ps(input_vec15, kern_value));
          __m128 input_vec16 = _mm_loadu_ps(padded_in_Offset2 + 60);
          sum_vec16 = _mm_add_ps(sum_vec16, _mm_mul_ps(input_vec16, kern_value));
          __m128 input_vec17 = _mm_loadu_ps(padded_in_Offset2 + 64);
          sum_vec17 = _mm_add_ps(sum_vec17, _mm_mul_ps(input_vec17, kern_value));
          __m128 input_vec18 = _mm_loadu_ps(padded_in_Offset2 + 68);
          sum_vec18 = _mm_add_ps(sum_vec18, _mm_mul_ps(input_vec18, kern_value));
          __m128 input_vec19 = _mm_loadu_ps(padded_in_Offset2 + 72);
          sum_vec19 = _mm_add_ps(sum_vec19, _mm_mul_ps(input_vec19, kern_value));
          __m128 input_vec20 = _mm_loadu_ps(padded_in_Offset2 + 76);
          sum_vec20 = _mm_add_ps(sum_vec20, _mm_mul_ps(input_vec20, kern_value));
          __m128 input_vec21 = _mm_loadu_ps(padded_in_Offset2 + 80);
          sum_vec21 = _mm_add_ps(sum_vec21, _mm_mul_ps(input_vec21, kern_value));
          __m128 input_vec22 = _mm_loadu_ps(padded_in_Offset2 + 84);
          sum_vec22 = _mm_add_ps(sum_vec22, _mm_mul_ps(input_vec22, kern_value));
          __m128 input_vec23 = _mm_loadu_ps(padded_in_Offset2 + 88);
          sum_vec23 = _mm_add_ps(sum_vec23, _mm_mul_ps(input_vec23, kern_value));
          __m128 input_vec24 = _mm_loadu_ps(padded_in_Offset2 + 92);	
          sum_vec24 = _mm_add_ps(sum_vec24, _mm_mul_ps(input_vec24, kern_value));	
          __m128 input_vec25 = _mm_loadu_ps(padded_in_Offset2 + 96);	
          sum_vec25 = _mm_add_ps(sum_vec25, _mm_mul_ps(input_vec25, kern_value));										
        }
      }	
      
      float *padded_out_ptr = padded_out + x + kern_cent_X + (y + kern_cent_Y) * padded_data_size_X;			
      _mm_storeu_ps(padded_out_ptr, sum_vec);
      _mm_storeu_ps(padded_out_ptr + 4, sum_vec2);
      _mm_storeu_ps(padded_out_ptr + 8, sum_vec3);
      _mm_storeu_ps(padded_out_ptr + 12, sum_vec4);		
      _mm_storeu_ps(padded_out_ptr + 16, sum_vec5);
      _mm_storeu_ps(padded_out_ptr + 20, sum_vec6);
      _mm_storeu_ps(padded_out_ptr + 24, sum_vec7);
      _mm_storeu_ps(padded_out_ptr + 28, sum_vec8);
      _mm_storeu_ps(padded_out_ptr + 32, sum_vec9);
      _mm_storeu_ps(padded_out_ptr + 36, sum_vec10);			
      _mm_storeu_ps(padded_out_ptr + 40, sum_vec11);
      _mm_storeu_ps(padded_out_ptr + 44, sum_vec12);
      _mm_storeu_ps(padded_out_ptr + 48, sum_vec13);
      _mm_storeu_ps(padded_out_ptr + 52, sum_vec14);		
      _mm_storeu_ps(padded_out_ptr + 56, sum_vec15);
      _mm_storeu_ps(padded_out_ptr + 60, sum_vec16);
      _mm_storeu_ps(padded_out_ptr + 64, sum_vec17);
      _mm_storeu_ps(padded_out_ptr + 68, sum_vec18);
      _mm_storeu_ps(padded_out_ptr + 72, sum_vec19);
      _mm_storeu_ps(padded_out_ptr + 76, sum_vec20);		
      _mm_storeu_ps(padded_out_ptr + 80, sum_vec21);
      _mm_storeu_ps(padded_out_ptr + 84, sum_vec22);
      _mm_storeu_ps(padded_out_ptr + 88, sum_vec23);
      _mm_storeu_ps(padded_out_ptr + 92, sum_vec24);		
      _mm_storeu_ps(padded_out_ptr + 96, sum_vec25);		
    }	
    
    int y_MUL_data_size_X;
    int y_PLUS_kern_MUL_pad;
#pragma omp parallel for
    for (int y = 0; y < data_size_Y; y++) {
      y_MUL_data_size_X = y * data_size_X;
      y_PLUS_kern_MUL_pad = (y + kern_cent_Y) * padded_data_size_X;
      for (int x = 400; x < data_size_X; x++) {
        out[x + y_MUL_data_size_X] =	padded_out[x + kern_cent_X + y_PLUS_kern_MUL_pad];
      }
    }     
  }	
	return 1;
}


