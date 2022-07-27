#ifndef _UNSHARP_MASK_HPP_
#define _UNSHARP_MASK_HPP_

#define __CL_ENABLE_EXCEPTIONS

#include "CL/cl.hpp"
#include "util.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>

//#include "blur.hpp"
#include "add_weighted.hpp"
#include "ppm.hpp"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#ifndef KERNEL_PATH
#define KERNEL_PATH "../kernel.cl"
#endif

void unsharp_mask(unsigned char *out, unsigned char *in,
                  int blur_radius, unsigned w, unsigned h, unsigned nchannels)
{
  const auto alpha = 1.5f, beta = -0.5f;
  std::vector<unsigned char> blur1, blur2, blur3, h_out;
  auto size = w * h * nchannels;

  blur1.resize(size);
  blur2.resize(size);
  blur3.resize(size);
  h_out.resize(size);

  try {
	  cl::Buffer d_in, d_blur1, d_blur2, d_blur3;

	  //// Create a context
	  cl::Context context(DEVICE);
	  cl::Program program(context, util::loadProgram(KERNEL_PATH), true);
	  cl::CommandQueue queue(context);

	  auto pixel_average = cl::make_kernel<cl::Buffer, cl::Buffer,
		 int, unsigned, unsigned, unsigned>(program, "pixel_average");

	  d_in	  =	cl::Buffer(context, in, in + size, true);
	  d_blur1 = cl::Buffer(context, std::begin(blur1), std::end(blur1), true);
	  d_blur2 = cl::Buffer(context, std::begin(blur2), std::end(blur2), true);
	  d_blur3 = cl::Buffer(context, std::begin(blur3), std::end(blur3), true);

	  pixel_average(cl::EnqueueArgs(queue, cl::NDRange(w,h)), d_blur1, d_in, blur_radius, w, h, nchannels);
	  pixel_average(cl::EnqueueArgs(queue, cl::NDRange(w,h)), d_blur2, d_blur1, blur_radius, w, h, nchannels);
	  pixel_average(cl::EnqueueArgs(queue, cl::NDRange(w,h)), d_blur3, d_blur2, blur_radius, w, h, nchannels);

	  queue.finish();

	  cl::copy(queue, d_blur3, begin(h_out), end(h_out));

	  add_weighted(out, in, alpha, h_out.data(), beta, 0.0f, w, h, nchannels);
  }
  catch (cl::Error error)
  {
	  std::cout << "Kernel error" << std::endl;
  }
}

#endif // _UNSHARP_MASK_HPP_
